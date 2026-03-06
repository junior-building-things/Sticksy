import json
import os
import re
import threading
import time
import html
import hashlib
import hmac
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
from zoneinfo import ZoneInfo

import requests
from flask import Flask, jsonify, request, Response
from google import genai
from google.genai import types
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError


app = Flask(__name__)

# -------- Environment --------

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
LARK_APP_ID = os.environ["LARK_APP_ID"]
LARK_APP_SECRET = os.environ["LARK_APP_SECRET"]
LARK_BOT_OPEN_ID = os.environ.get("LARK_BOT_OPEN_ID", "")
LARK_VERIFICATION_TOKEN = os.environ.get("LARK_VERIFICATION_TOKEN", "")
LARK_REQUEST_SIGNING_SECRET = os.environ.get("LARK_REQUEST_SIGNING_SECRET", "")
KLIPY_API_KEY = os.environ.get("KLIPY_API_KEY", "")
KLIPY_SEARCH_URL = os.environ.get("KLIPY_SEARCH_URL", "https://api.klipy.com/v1/stickers/search")
THOMAS_OPEN_ID = os.environ.get("THOMAS_OPEN_ID", "")
THOMAS_DISPLAY_NAME = os.environ.get("THOMAS_DISPLAY_NAME", "Thomas")
RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "30"))
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")
DATABASE_PATH = os.environ.get("DATABASE_PATH", "sticksy.db")
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
if not DATABASE_URL:
    DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

HTTP_TIMEOUT = 12
MAX_SUMMARY_IMAGES = 3
MAX_SUMMARY_MESSAGES = 350
TOPIC_WINDOW_HOURS = 6
MAX_STICKER_UPLOAD_BYTES = int(os.environ.get("MAX_STICKER_UPLOAD_BYTES", "9500000"))
MAX_MEETING_TRANSCRIPT_CHARS = int(os.environ.get("MAX_MEETING_TRANSCRIPT_CHARS", "120000"))
MAX_MEETING_MEDIA_BYTES = int(os.environ.get("MAX_MEETING_MEDIA_BYTES", "18000000"))
SUMMARY_REWRITE_LOOKBACK_SECONDS = int(os.environ.get("SUMMARY_REWRITE_LOOKBACK_SECONDS", "1800"))
GLOBAL_LEARN_SCOPE = "__global__"

client = genai.Client(api_key=GEMINI_API_KEY)

_tenant_token_cache = {"token": None, "expires_at": 0}
_db_lock = threading.Lock()
_last_retention_sweep = 0.0


# -------- Prompts --------

SUMMARY_SYSTEM_PROMPT = """
You are Sticksy, a concise group-chat summarizer.
Rules:
- Preserve the language of the user's prompt.
- Output JSON only.
- Focus on important topics. If many topics exist, keep lower-priority topics very short.
- Attribute claims to specific speaker names when possible.
- Use any provided learned terminology exactly when relevant.
- Always produce a summary, even when content is sparse.
- Synthesize and compress; do not copy chat lines verbatim.

JSON schema:
{
  "intro": "string",
  "topics": [
    {"title":"string","summary":"string"}
  ],
  "important_image_keys": ["string"]
}

Constraints:
- intro should be a brief synthesized overview (not a quote).
- topics should be ordered by importance.
- each topic summary must be a synthesized takeaway, not a transcript line.
- include up to 3 image keys from the provided image pool if they are relevant.
""".strip()

SUMMARY_EDIT_SYSTEM_PROMPT = """
You are Sticksy, editing an existing summary according to the user's instruction.
Return JSON only: {"edited_summary":"string"}.
Rules:
- Preserve the original language and structure unless the instruction requires a structural change.
- Apply only the requested edit; keep all unaffected content intact.
- Do not add new facts.
- If the instruction is ambiguous, make the smallest reasonable edit.
- If the instruction cannot be applied, return the original summary unchanged.
""".strip()

TOPIC_SYSTEM_PROMPT = """
Extract up to 8 current discussion topics from recent group messages.
Return JSON only: {"topics":[{"topic":"string","importance":1-5}]}
Keep topics short and concrete.
""".strip()

STICKER_QUERY_SYSTEM_PROMPT = """
Generate a concise sticker search query from the user's message. The query will be used to retrieve relevant reply stickers.
Return JSON only: {"query":"..."}.
Rules:
- Use the recent chat context to infer tone and intent before generating the query.
- The query should serve as a fun and cheeky reply to the user's message. Add a playful twist where possible.
- Preserve the user's language when possible.
- Keep most queries 1-2 words. Avoid more than 3 unless it's a specific term (e.g. kpop demon hunters).
""".strip()

MEETING_SUMMARY_SYSTEM_PROMPT = """
You are Sticksy, a concise meeting summarizer.
Return JSON only with this schema:
{
  "summary_bullets": ["string"],
  "next_steps": [
    {"owner_name":"string","owner_email":"string","task":"string"}
  ]
}

Rules:
- Preserve the language of the user's request.
- Use the meeting transcript or meeting media as the source of truth.
- Prefer owner names that exactly match the provided known participant names when applicable.
- When an owner matches a provided owner candidate, copy the owner name exactly from owner_candidates.
- Never translate, localize, or romanize people's names. Preserve person names exactly as they appear in the transcript or known_people.
- If the transcript includes a speaker email, include it in owner_email.
- Use any provided learned terminology exactly when relevant.
- Always synthesize; do not copy long transcript lines verbatim.
- summary_bullets should be concise key takeaways with no labels.
- next_steps should capture owner-specific action items when the owner is clear.
- If an owner is unclear, leave owner_name empty rather than guessing.
- Keep next steps concrete and brief.
""".strip()


class MeetingTranscriptAccessError(RuntimeError):
    pass


# -------- DB --------

ENGINE = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    future=True,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
IS_POSTGRES = ENGINE.dialect.name.startswith("postgres")


def db_execute(query: str, params: dict | None = None):
    with _db_lock:
        with ENGINE.begin() as conn:
            conn.execute(text(query), params or {})


def db_query_all(query: str, params: dict | None = None) -> list[dict]:
    with _db_lock:
        with ENGINE.connect() as conn:
            rows = conn.execute(text(query), params or {}).mappings().all()
            return [dict(r) for r in rows]


def db_query_one(query: str, params: dict | None = None) -> dict | None:
    with _db_lock:
        with ENGINE.connect() as conn:
            row = conn.execute(text(query), params or {}).mappings().first()
            return dict(row) if row else None


def init_db():
    if IS_POSTGRES:
        schema_sql = [
            """
            CREATE TABLE IF NOT EXISTS messages (
              id BIGSERIAL PRIMARY KEY,
              event_message_id TEXT,
              chat_id TEXT NOT NULL,
              chat_name TEXT,
              sender_open_id TEXT,
              sender_name TEXT,
              is_from_bot INTEGER NOT NULL DEFAULT 0,
              message_type TEXT NOT NULL,
              text_content TEXT,
              image_key TEXT,
              root_id TEXT,
              parent_id TEXT,
              created_at_ts BIGINT NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_messages_chat_time ON messages(chat_id, created_at_ts DESC)",
            "CREATE INDEX IF NOT EXISTS idx_messages_root_time ON messages(root_id, created_at_ts DESC)",
            """
            CREATE TABLE IF NOT EXISTS processed_events (
              event_id TEXT PRIMARY KEY,
              created_at_ts BIGINT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS bot_messages (
              message_id TEXT PRIMARY KEY,
              chat_id TEXT NOT NULL,
              created_at_ts BIGINT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS topic_cache (
              chat_id TEXT PRIMARY KEY,
              topics_json TEXT NOT NULL,
              computed_at_ts BIGINT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_timezone_cache (
              open_id TEXT PRIMARY KEY,
              tz_name TEXT NOT NULL,
              cached_at_ts BIGINT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_profile_cache (
              open_id TEXT PRIMARY KEY,
              tz_name TEXT NOT NULL,
              display_name TEXT NOT NULL,
              cached_at_ts BIGINT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chat_profile_cache (
              chat_id TEXT PRIMARY KEY,
              chat_name TEXT NOT NULL,
              cached_at_ts BIGINT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS learned_terms (
              chat_id TEXT NOT NULL,
              normalized_key TEXT NOT NULL,
              instruction_text TEXT NOT NULL,
              learned_by TEXT,
              created_at_ts BIGINT NOT NULL,
              PRIMARY KEY (chat_id, normalized_key)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS mentioned_identities (
              chat_id TEXT NOT NULL,
              normalized_name TEXT NOT NULL,
              open_id TEXT NOT NULL,
              display_name TEXT NOT NULL,
              created_at_ts BIGINT NOT NULL,
              PRIMARY KEY (chat_id, normalized_name, open_id)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_mentioned_identities_chat_time ON mentioned_identities(chat_id, created_at_ts DESC)",
            """
            CREATE TABLE IF NOT EXISTS meeting_summary_cache (
              minute_token TEXT NOT NULL,
              mode TEXT NOT NULL,
              reply_text TEXT NOT NULL,
              title TEXT NOT NULL,
              source_chat_id TEXT,
              created_at_ts BIGINT NOT NULL,
              PRIMARY KEY (minute_token, mode)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_meeting_summary_cache_created ON meeting_summary_cache(created_at_ts DESC)",
        ]
    else:
        schema_sql = [
            """
            CREATE TABLE IF NOT EXISTS messages (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              event_message_id TEXT,
              chat_id TEXT NOT NULL,
              chat_name TEXT,
              sender_open_id TEXT,
              sender_name TEXT,
              is_from_bot INTEGER NOT NULL DEFAULT 0,
              message_type TEXT NOT NULL,
              text_content TEXT,
              image_key TEXT,
              root_id TEXT,
              parent_id TEXT,
              created_at_ts INTEGER NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_messages_chat_time ON messages(chat_id, created_at_ts DESC)",
            "CREATE INDEX IF NOT EXISTS idx_messages_root_time ON messages(root_id, created_at_ts DESC)",
            """
            CREATE TABLE IF NOT EXISTS processed_events (
              event_id TEXT PRIMARY KEY,
              created_at_ts INTEGER NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS bot_messages (
              message_id TEXT PRIMARY KEY,
              chat_id TEXT NOT NULL,
              created_at_ts INTEGER NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS topic_cache (
              chat_id TEXT PRIMARY KEY,
              topics_json TEXT NOT NULL,
              computed_at_ts INTEGER NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_timezone_cache (
              open_id TEXT PRIMARY KEY,
              tz_name TEXT NOT NULL,
              cached_at_ts INTEGER NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_profile_cache (
              open_id TEXT PRIMARY KEY,
              tz_name TEXT NOT NULL,
              display_name TEXT NOT NULL,
              cached_at_ts INTEGER NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chat_profile_cache (
              chat_id TEXT PRIMARY KEY,
              chat_name TEXT NOT NULL,
              cached_at_ts INTEGER NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS learned_terms (
              chat_id TEXT NOT NULL,
              normalized_key TEXT NOT NULL,
              instruction_text TEXT NOT NULL,
              learned_by TEXT,
              created_at_ts INTEGER NOT NULL,
              PRIMARY KEY (chat_id, normalized_key)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS mentioned_identities (
              chat_id TEXT NOT NULL,
              normalized_name TEXT NOT NULL,
              open_id TEXT NOT NULL,
              display_name TEXT NOT NULL,
              created_at_ts INTEGER NOT NULL,
              PRIMARY KEY (chat_id, normalized_name, open_id)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_mentioned_identities_chat_time ON mentioned_identities(chat_id, created_at_ts DESC)",
            """
            CREATE TABLE IF NOT EXISTS meeting_summary_cache (
              minute_token TEXT NOT NULL,
              mode TEXT NOT NULL,
              reply_text TEXT NOT NULL,
              title TEXT NOT NULL,
              source_chat_id TEXT,
              created_at_ts INTEGER NOT NULL,
              PRIMARY KEY (minute_token, mode)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_meeting_summary_cache_created ON meeting_summary_cache(created_at_ts DESC)",
        ]

    for stmt in schema_sql:
        db_execute(stmt)


init_db()


# -------- Utility --------


def now_ts() -> int:
    return int(time.time())


def maybe_sweep_retention():
    global _last_retention_sweep
    current = time.time()
    if current - _last_retention_sweep < 3600:
        return
    cutoff = now_ts() - RETENTION_DAYS * 24 * 3600
    db_execute("DELETE FROM messages WHERE created_at_ts < :cutoff", {"cutoff": cutoff})
    db_execute("DELETE FROM bot_messages WHERE created_at_ts < :cutoff", {"cutoff": cutoff})
    db_execute("DELETE FROM processed_events WHERE created_at_ts < :cutoff", {"cutoff": cutoff})
    db_execute("DELETE FROM topic_cache WHERE computed_at_ts < :cutoff", {"cutoff": cutoff})
    db_execute("DELETE FROM user_timezone_cache WHERE cached_at_ts < :cutoff", {"cutoff": cutoff})
    db_execute("DELETE FROM user_profile_cache WHERE cached_at_ts < :cutoff", {"cutoff": cutoff})
    db_execute("DELETE FROM chat_profile_cache WHERE cached_at_ts < :cutoff", {"cutoff": cutoff})
    db_execute("DELETE FROM meeting_summary_cache WHERE created_at_ts < :cutoff", {"cutoff": cutoff})
    _last_retention_sweep = current


def normalize_learning_key(text_value: str) -> str:
    return re.sub(r"\s+", " ", (text_value or "").strip().lower())


def parse_learning_instruction(text_value: str) -> str:
    match = re.match(r"^\s*learn(?:\s+|:\s*)(.+?)\s*$", text_value or "", re.IGNORECASE)
    if not match:
        return ""
    return sanitize_text(match.group(1) or "")


def save_learned_term(chat_id: str, instruction_text: str, learned_by: str = ""):
    normalized_key = normalize_learning_key(instruction_text)
    if not normalized_key:
        return

    params = {
        "chat_id": GLOBAL_LEARN_SCOPE,
        "normalized_key": normalized_key,
        "instruction_text": instruction_text.strip(),
        "learned_by": (learned_by or "").strip(),
        "created_at_ts": now_ts(),
    }
    if IS_POSTGRES:
        db_execute(
            """
            INSERT INTO learned_terms(chat_id, normalized_key, instruction_text, learned_by, created_at_ts)
            VALUES(:chat_id, :normalized_key, :instruction_text, :learned_by, :created_at_ts)
            ON CONFLICT (chat_id, normalized_key) DO UPDATE
            SET instruction_text = EXCLUDED.instruction_text,
                learned_by = EXCLUDED.learned_by,
                created_at_ts = EXCLUDED.created_at_ts
            """,
            params,
        )
    else:
        db_execute(
            """
            INSERT OR REPLACE INTO learned_terms(chat_id, normalized_key, instruction_text, learned_by, created_at_ts)
            VALUES(:chat_id, :normalized_key, :instruction_text, :learned_by, :created_at_ts)
            """,
            params,
        )


def load_learned_terms(chat_id: str, limit: int = 40) -> list[str]:
    rows = db_query_all(
        """
        SELECT instruction_text
        FROM learned_terms
        WHERE chat_id = :chat_id
        ORDER BY created_at_ts DESC
        LIMIT :limit_rows
        """,
        {"chat_id": GLOBAL_LEARN_SCOPE, "limit_rows": limit},
    )
    return [(row.get("instruction_text") or "").strip() for row in rows if (row.get("instruction_text") or "").strip()]


def clean_term_candidate(value: str) -> str:
    cleaned = (value or "").strip()
    cleaned = cleaned.strip("`'\"“”‘’.,;:()[]{}<>")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def extract_learning_replacements(instruction_text: str) -> list[tuple[str, str]]:
    cleaned = sanitize_text(instruction_text or "")
    if not cleaned:
        return []
    body = re.sub(r"^\s*that\s+", "", cleaned, flags=re.IGNORECASE).strip()

    canonical = ""
    parenthetical = ""
    match = re.match(r"^\s*([^()]{1,100}?)\s*\(([^)]{1,140})\)\s*(?:refers to|means|is)\b", body, re.IGNORECASE)
    if match:
        canonical = clean_term_candidate(match.group(1))
        parenthetical = match.group(2)
    else:
        match = re.match(r"^\s*([^()]{1,100}?)\s*(?:refers to|means|is)\b", body, re.IGNORECASE)
        if match:
            canonical = clean_term_candidate(match.group(1))

    if not canonical:
        return []

    alias_candidates: list[str] = []
    capture_pattern = r"(?:transcribed as|also transcribed as|sometimes transcribed as|aka|also called|written as)\s+([A-Za-z0-9][A-Za-z0-9\-/]{1,40})"
    if parenthetical:
        alias_candidates.extend(re.findall(capture_pattern, parenthetical, flags=re.IGNORECASE))
        alias_candidates.extend(re.findall(r"\b[A-Za-z][A-Za-z0-9\-/]{1,40}\b", parenthetical))
    alias_candidates.extend(re.findall(capture_pattern, body, flags=re.IGNORECASE))

    stopwords = {"sometimes", "transcribed", "as", "also", "written", "called", "aka", "or", "and", "the", "that"}
    canonical_key = normalize_learning_key(canonical)
    replacements: list[tuple[str, str]] = []
    seen: set[str] = set()
    for alias in alias_candidates:
        clean_alias = clean_term_candidate(alias)
        alias_key = normalize_learning_key(clean_alias)
        if not clean_alias or alias_key in seen or alias_key in stopwords:
            continue
        if alias_key == canonical_key:
            continue
        seen.add(alias_key)
        replacements.append((clean_alias, canonical))
    return replacements


def replace_term_occurrences(text_value: str, source_term: str, target_term: str) -> str:
    source = clean_term_candidate(source_term)
    target = clean_term_candidate(target_term)
    if not source or not target or normalize_learning_key(source) == normalize_learning_key(target):
        return text_value
    pattern = re.compile(
        rf"(?<![0-9A-Za-z\u4e00-\u9fff]){re.escape(source)}(?![0-9A-Za-z\u4e00-\u9fff])",
        flags=re.IGNORECASE,
    )
    return pattern.sub(target, text_value or "")


def apply_learning_replacements(text_value: str, instructions: list[str]) -> str:
    if not text_value:
        return ""
    replacement_map: dict[str, tuple[str, str]] = {}
    for instruction in instructions:
        for source_term, target_term in extract_learning_replacements(instruction):
            source_key = normalize_learning_key(source_term)
            if source_key and source_key not in replacement_map:
                replacement_map[source_key] = (source_term, target_term)

    rewritten = text_value
    ordered_pairs = sorted(replacement_map.values(), key=lambda pair: len(pair[0]), reverse=True)
    for source_term, target_term in ordered_pairs:
        rewritten = replace_term_occurrences(rewritten, source_term, target_term)
    return rewritten


def looks_like_summary_output(text_value: str) -> bool:
    lowered = (text_value or "").strip().lower()
    if not lowered:
        return False
    if lowered.startswith("summary of "):
        return True
    if "meeting summary:" in lowered or "<u>meeting summary:</u>" in lowered:
        return True
    if lowered.startswith("next steps:") or "<u>next steps:</u>" in lowered:
        return True
    return False


def load_recent_bot_texts(chat_id: str, cutoff_ts: int, root_id: str | None = None, limit: int = 40) -> list[dict]:
    params = {
        "chat_id": chat_id,
        "cutoff_ts": cutoff_ts,
        "limit_rows": limit,
    }
    sql = """
        SELECT text_content
        FROM messages
        WHERE chat_id = :chat_id
          AND is_from_bot = 1
          AND message_type = 'text'
          AND text_content IS NOT NULL
          AND text_content <> ''
          AND created_at_ts >= :cutoff_ts
    """
    if root_id:
        sql += " AND root_id = :root_id"
        params["root_id"] = root_id
    sql += " ORDER BY created_at_ts DESC LIMIT :limit_rows"
    return db_query_all(sql, params)


def latest_summary_output(chat_id: str, root_id: str | None = None) -> str:
    cutoff_ts = now_ts() - max(60, SUMMARY_REWRITE_LOOKBACK_SECONDS)
    scopes = [root_id, None] if root_id else [None]
    for scoped_root in scopes:
        rows = load_recent_bot_texts(chat_id, cutoff_ts, root_id=scoped_root, limit=40)
        for row in rows:
            text_value = (row.get("text_content") or "").strip()
            if looks_like_summary_output(text_value):
                return text_value
    return ""


def rewrite_latest_summary_after_learning(chat_id: str, root_id: str | None, learning_text: str) -> str:
    latest_summary = latest_summary_output(chat_id, root_id=root_id)
    if not latest_summary:
        return ""

    rewritten = apply_learning_replacements(latest_summary, [learning_text])
    if rewritten != latest_summary:
        return rewritten

    all_terms = load_learned_terms(chat_id)
    rewritten = apply_learning_replacements(latest_summary, all_terms)
    if rewritten != latest_summary:
        return rewritten
    return ""


def parse_summary_edit_instruction(text_value: str) -> str:
    match = re.match(r"^\s*edit\s+summary(?:\s*:\s*|\s+)(.+?)\s*$", text_value or "", re.IGNORECASE)
    if not match:
        return ""
    return sanitize_text(match.group(1) or "")


def parse_ordinal_position(text_value: str) -> int:
    lowered = (text_value or "").lower()
    word_map = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
        "tenth": 10,
        "last": -1,
    }
    for word, value in word_map.items():
        if re.search(rf"\b{word}\b", lowered):
            return value
    match = re.search(r"\b(\d+)(?:st|nd|rd|th)?\b", lowered)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return 0
    return 0


def summary_section_name(line: str) -> str:
    clean = re.sub(r"<[^>]+>", "", line or "").strip().lower()
    clean = re.sub(r"\s+", " ", clean).rstrip(":")
    if clean in {"meeting summary", "next steps"}:
        return clean
    return ""


def section_range(lines: list[str], target_section: str) -> tuple[int, int]:
    start_idx = -1
    for idx, line in enumerate(lines):
        if summary_section_name(line) == target_section:
            start_idx = idx + 1
            break
    if start_idx < 0:
        return -1, -1

    end_idx = len(lines)
    for idx in range(start_idx, len(lines)):
        name = summary_section_name(lines[idx])
        if name and name != target_section:
            end_idx = idx
            break
    return start_idx, end_idx


def renumber_numbered_lines(lines: list[str], start_idx: int, end_idx: int):
    counter = 1
    for idx in range(max(0, start_idx), min(len(lines), end_idx)):
        match = re.match(r"^(\s*)\d+\.\s+(.*)$", lines[idx])
        if not match:
            continue
        lines[idx] = f"{match.group(1)}{counter}. {match.group(2)}"
        counter += 1


def remove_nth_item_from_section(text_value: str, section_name: str, position: int) -> str:
    if not text_value or position == 0:
        return text_value
    lines = (text_value or "").splitlines()
    start_idx, end_idx = section_range(lines, section_name)
    if start_idx < 0:
        return text_value

    item_indices = []
    for idx in range(start_idx, end_idx):
        if re.match(r"^\s*(?:[-*]|\d+\.)\s+\S", lines[idx]):
            item_indices.append(idx)
    if not item_indices:
        return text_value

    if position < 0:
        remove_idx = item_indices[-1]
    elif 0 < position <= len(item_indices):
        remove_idx = item_indices[position - 1]
    else:
        return text_value

    del lines[remove_idx]
    if remove_idx < end_idx:
        end_idx -= 1
    renumber_numbered_lines(lines, start_idx, end_idx)
    edited = "\n".join(lines)
    edited = re.sub(r"\n{3,}", "\n\n", edited).strip()
    return edited


def remove_nth_item_from_any_list(text_value: str, position: int) -> str:
    if not text_value or position == 0:
        return text_value
    lines = (text_value or "").splitlines()
    item_indices = [idx for idx, line in enumerate(lines) if re.match(r"^\s*(?:[-*]|\d+\.)\s+\S", line)]
    if not item_indices:
        return text_value

    if position < 0:
        remove_idx = item_indices[-1]
    elif 0 < position <= len(item_indices):
        remove_idx = item_indices[position - 1]
    else:
        return text_value

    del lines[remove_idx]
    for section in ["meeting summary", "next steps"]:
        start_idx, end_idx = section_range(lines, section)
        if start_idx >= 0:
            renumber_numbered_lines(lines, start_idx, end_idx)
    edited = "\n".join(lines)
    edited = re.sub(r"\n{3,}", "\n\n", edited).strip()
    return edited


def apply_structured_summary_edit(summary_text: str, instruction: str) -> str:
    lowered = (instruction or "").lower()
    if not lowered:
        return summary_text

    remove_intent = any(token in lowered for token in ["remove", "delete", "drop"])
    if not remove_intent:
        return summary_text

    position = parse_ordinal_position(lowered)
    if position == 0:
        position = -1 if "last" in lowered else 1

    if "next step" in lowered or "action item" in lowered or "todo" in lowered:
        edited = remove_nth_item_from_section(summary_text, "next steps", position)
        if edited != summary_text:
            return edited

    if "meeting summary" in lowered or "summary bullet" in lowered or "summary point" in lowered:
        edited = remove_nth_item_from_section(summary_text, "meeting summary", position)
        if edited != summary_text:
            return edited

    if any(token in lowered for token in ["bullet", "point", "item", "line", "step"]):
        edited = remove_nth_item_from_any_list(summary_text, position)
        if edited != summary_text:
            return edited

    return summary_text


def edit_summary_with_gemini(chat_id: str, summary_text: str, instruction: str) -> str:
    user_block = {
        "instruction": instruction,
        "summary": summary_text,
        "learned_terms": load_learned_terms(chat_id),
    }
    config = types.GenerateContentConfig(
        system_instruction=SUMMARY_EDIT_SYSTEM_PROMPT,
        response_mime_type="application/json",
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=json.dumps(user_block, ensure_ascii=False),
        config=config,
    )
    raw = (response.text or "").strip()
    parsed = json.loads(raw)
    return (parsed.get("edited_summary") or "").strip()


def edit_latest_summary(chat_id: str, root_id: str | None, instruction: str) -> tuple[str, str]:
    latest_summary = latest_summary_output(chat_id, root_id=root_id)
    if not latest_summary:
        return "", ""

    edited = apply_structured_summary_edit(latest_summary, instruction)
    if edited and edited != latest_summary:
        return latest_summary, edited

    try:
        model_edited = edit_summary_with_gemini(chat_id, latest_summary, instruction)
        if model_edited and model_edited != latest_summary:
            return latest_summary, model_edited
    except Exception:
        app.logger.exception("Summary edit via Gemini failed")

    return latest_summary, ""


def save_mentioned_identity(chat_id: str, open_id: str, display_name: str):
    clean_chat_id = (chat_id or "").strip()
    clean_open_id = (open_id or "").strip()
    clean_name = (display_name or "").strip()
    normalized_name = normalize_person_name(clean_name)
    if not clean_chat_id or not clean_open_id or not clean_name or not normalized_name:
        return

    params = {
        "chat_id": clean_chat_id,
        "normalized_name": normalized_name,
        "open_id": clean_open_id,
        "display_name": clean_name,
        "created_at_ts": now_ts(),
    }
    if IS_POSTGRES:
        db_execute(
            """
            INSERT INTO mentioned_identities(chat_id, normalized_name, open_id, display_name, created_at_ts)
            VALUES(:chat_id, :normalized_name, :open_id, :display_name, :created_at_ts)
            ON CONFLICT (chat_id, normalized_name, open_id) DO UPDATE
            SET display_name = EXCLUDED.display_name,
                created_at_ts = EXCLUDED.created_at_ts
            """,
            params,
        )
    else:
        db_execute(
            """
            INSERT OR REPLACE INTO mentioned_identities(chat_id, normalized_name, open_id, display_name, created_at_ts)
            VALUES(:chat_id, :normalized_name, :open_id, :display_name, :created_at_ts)
            """,
            params,
        )


def recent_mentioned_identities(chat_id: str | None = None, limit: int = 300) -> list[dict]:
    params = {"limit_rows": limit}
    if chat_id:
        return db_query_all(
            """
            SELECT open_id, display_name, MAX(created_at_ts) AS last_seen_ts
            FROM mentioned_identities
            WHERE chat_id = :chat_id
            GROUP BY open_id, display_name
            ORDER BY last_seen_ts DESC
            LIMIT :limit_rows
            """,
            {"chat_id": chat_id, "limit_rows": limit},
        )

    return db_query_all(
        """
        SELECT open_id, display_name, MAX(created_at_ts) AS last_seen_ts
        FROM mentioned_identities
        GROUP BY open_id, display_name
        ORDER BY last_seen_ts DESC
        LIMIT :limit_rows
        """,
        params,
    )


def verify_lark_request(raw_body: bytes, payload: dict) -> tuple[bool, str]:
    if LARK_VERIFICATION_TOKEN and payload.get("token") != LARK_VERIFICATION_TOKEN:
        return False, "token mismatch"

    if LARK_REQUEST_SIGNING_SECRET:
        ts = request.headers.get("X-Lark-Request-Timestamp")
        sig = request.headers.get("X-Lark-Signature")
        if not ts or not sig:
            return False, "missing signature headers"

        try:
            ts_int = int(ts)
        except ValueError:
            return False, "invalid timestamp"

        if abs(now_ts() - ts_int) > 300:
            return False, "stale request"

        raw_text = raw_body.decode("utf-8")
        expected = hmac.new(
            LARK_REQUEST_SIGNING_SECRET.encode("utf-8"),
            msg=(ts + raw_text).encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, sig):
            return False, "signature mismatch"

    return True, ""


def record_event_once(event_id: str | None) -> bool:
    if not event_id:
        return True
    try:
        db_execute(
            "INSERT INTO processed_events(event_id, created_at_ts) VALUES(:event_id, :created_at_ts)",
            {"event_id": event_id, "created_at_ts": now_ts()},
        )
        return True
    except IntegrityError:
        return False


def get_tenant_access_token() -> str:
    current = now_ts()
    if _tenant_token_cache["token"] and current < _tenant_token_cache["expires_at"]:
        return _tenant_token_cache["token"]

    resp = requests.post(
        "https://open.larkoffice.com/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": LARK_APP_ID, "app_secret": LARK_APP_SECRET},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    body = resp.json()
    token = body["tenant_access_token"]
    ttl = int(body.get("expire", 3600))
    _tenant_token_cache["token"] = token
    _tenant_token_cache["expires_at"] = current + max(60, ttl - 60)
    return token


def lark_headers() -> dict:
    return {
        "Authorization": f"Bearer {get_tenant_access_token()}",
        "Content-Type": "application/json",
    }


def send_lark_text_reply(reply_to_message_id: str, text: str, mention_open_id: str | None = None, mention_name: str | None = None) -> str | None:
    prefix = ""
    if mention_open_id:
        display = mention_name or "there"
        prefix = f'<at user_id="{mention_open_id}">{display}</at> '

    payload = {
        "msg_type": "text",
        "content": json.dumps({"text": f"{prefix}{text}".strip()}),
    }
    resp = requests.post(
        f"https://open.larkoffice.com/open-apis/im/v1/messages/{reply_to_message_id}/reply",
        headers=lark_headers(),
        json=payload,
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return ((data.get("data") or {}).get("message_id"))


def build_post_content_from_text(text: str) -> tuple[str, list[list[dict]]]:
    lines = (text or "").splitlines()
    content: list[list[dict]] = []
    at_pattern = re.compile(r'<at user_id="([^"]+)">(.+?)</at>', re.IGNORECASE)
    underlined_headers = {"Meeting summary:", "Next steps:"}

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip():
            if content:
                content.append([{"tag": "text", "text": " "}])
            continue
        if line.strip() in underlined_headers:
            content.append([{"tag": "text", "text": line.strip(), "style": ["underline"]}])
            continue

        paragraph: list[dict] = []
        cursor = 0
        for match in at_pattern.finditer(line):
            prefix = line[cursor:match.start()]
            if prefix:
                paragraph.append({"tag": "text", "text": prefix})
            paragraph.append(
                {
                    "tag": "at",
                    "user_id": (match.group(1) or "").strip(),
                    "user_name": (match.group(2) or "").strip() or "there",
                }
            )
            cursor = match.end()
        suffix = line[cursor:]
        if suffix:
            paragraph.append({"tag": "text", "text": suffix})
        if not paragraph:
            paragraph = [{"tag": "text", "text": line}]
        content.append(paragraph)

    if not content:
        content = [[{"tag": "text", "text": text or "Sticksy"}]]
    return "", content


def send_lark_post_reply(
    reply_to_message_id: str,
    text: str,
    title: str = "",
    mention_open_id: str | None = None,
    mention_name: str | None = None,
) -> str | None:
    _, content = build_post_content_from_text(text)
    if mention_open_id:
        mention_parts = [
            {"tag": "at", "user_id": mention_open_id, "user_name": mention_name or "there"},
            {"tag": "text", "text": " "},
        ]
        if content:
            content[0] = mention_parts + content[0]
        else:
            content = [mention_parts]
    locale_key = "zh_cn" if looks_like_cjk(text) else "en_us"
    post_body = {
        locale_key: {
            "title": sanitize_text(title),
            "content": content,
        }
    }
    payload = {
        "msg_type": "post",
        "content": json.dumps(post_body, ensure_ascii=False),
    }
    resp = requests.post(
        f"https://open.larkoffice.com/open-apis/im/v1/messages/{reply_to_message_id}/reply",
        headers=lark_headers(),
        json=payload,
        timeout=HTTP_TIMEOUT,
    )
    if resp.status_code >= 400:
        app.logger.warning("Lark post reply failed status=%s body=%s", resp.status_code, resp.text[:500])
        raise RuntimeError(f"lark post reply failed status={resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    return ((data.get("data") or {}).get("message_id"))


def send_lark_image_reply(reply_to_message_id: str, image_key: str) -> str | None:
    payload = {
        "msg_type": "image",
        "content": json.dumps({"image_key": image_key}),
    }
    resp = requests.post(
        f"https://open.larkoffice.com/open-apis/im/v1/messages/{reply_to_message_id}/reply",
        headers=lark_headers(),
        json=payload,
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return ((data.get("data") or {}).get("message_id"))


def upload_lark_image(image_bytes: bytes, filename: str = "sticker.gif") -> str | None:
    headers = {"Authorization": f"Bearer {get_tenant_access_token()}"}
    files = {
        "image_type": (None, "message"),
        "image": (filename, image_bytes),
    }
    resp = requests.post(
        "https://open.larkoffice.com/open-apis/im/v1/images",
        headers=headers,
        files=files,
        timeout=HTTP_TIMEOUT,
    )
    if resp.status_code >= 400:
        app.logger.warning("Lark image upload HTTP error status=%s body=%s", resp.status_code, resp.text[:300])
        return None
    body = resp.json()
    if body.get("code", 0) != 0:
        app.logger.warning("Lark image upload API error code=%s msg=%s", body.get("code"), body.get("msg"))
        return None
    return (body.get("data") or {}).get("image_key")


def remember_bot_message(message_id: str | None, chat_id: str):
    if not message_id:
        return
    if IS_POSTGRES:
        db_execute(
            """
            INSERT INTO bot_messages(message_id, chat_id, created_at_ts)
            VALUES(:message_id, :chat_id, :created_at_ts)
            ON CONFLICT (message_id) DO UPDATE
            SET chat_id = EXCLUDED.chat_id, created_at_ts = EXCLUDED.created_at_ts
            """,
            {"message_id": message_id, "chat_id": chat_id, "created_at_ts": now_ts()},
        )
    else:
        db_execute(
            "INSERT OR REPLACE INTO bot_messages(message_id, chat_id, created_at_ts) VALUES(:message_id, :chat_id, :created_at_ts)",
            {"message_id": message_id, "chat_id": chat_id, "created_at_ts": now_ts()},
        )


def save_incoming_message(
    *,
    event_message_id: str,
    chat_id: str,
    chat_name: str,
    sender_open_id: str,
    sender_name: str,
    message_type: str,
    text_content: str,
    image_key: str,
    root_id: str,
    parent_id: str,
    created_at_ts: int,
):
    db_execute(
        """
        INSERT INTO messages(
          event_message_id, chat_id, chat_name, sender_open_id, sender_name, is_from_bot,
          message_type, text_content, image_key, root_id, parent_id, created_at_ts
        ) VALUES (:event_message_id, :chat_id, :chat_name, :sender_open_id, :sender_name, 0,
        :message_type, :text_content, :image_key, :root_id, :parent_id, :created_at_ts)
        """,
        {
            "event_message_id": event_message_id,
            "chat_id": chat_id,
            "chat_name": chat_name,
            "sender_open_id": sender_open_id,
            "sender_name": sender_name,
            "message_type": message_type,
            "text_content": text_content,
            "image_key": image_key,
            "root_id": root_id,
            "parent_id": parent_id,
            "created_at_ts": created_at_ts,
        },
    )


def save_bot_text(chat_id: str, text: str, root_id: str | None, parent_id: str | None, event_message_id: str | None = None):
    db_execute(
        """
        INSERT INTO messages(
          event_message_id, chat_id, chat_name, sender_open_id, sender_name, is_from_bot,
          message_type, text_content, image_key, root_id, parent_id, created_at_ts
        ) VALUES(:event_message_id, :chat_id, '', '', 'Sticksy', 1, 'text', :text, '', :root_id, :parent_id, :created_at_ts)
        """,
        {
            "event_message_id": event_message_id,
            "chat_id": chat_id,
            "text": text,
            "root_id": root_id or "",
            "parent_id": parent_id or "",
            "created_at_ts": now_ts(),
        },
    )


def is_reply_to_bot(parent_id: str | None, root_id: str | None) -> bool:
    if not parent_id and not root_id:
        return False
    row = db_query_one(
        "SELECT 1 FROM bot_messages WHERE message_id = :parent_id OR message_id = :root_id LIMIT 1",
        {"parent_id": parent_id or "", "root_id": root_id or ""},
    )
    if row is not None:
        return True

    row2 = db_query_one(
        """
        SELECT 1
        FROM messages
        WHERE is_from_bot = 1 AND (event_message_id = :parent_id OR event_message_id = :root_id)
        LIMIT 1
        """,
        {"parent_id": parent_id or "", "root_id": root_id or ""},
    )
    if row2 is not None:
        return True

    # Fallback: check Lark message metadata directly in case local cache missed a bot message id.
    for message_id in [parent_id, root_id]:
        if message_id and lark_message_is_from_bot(message_id):
            return True
    return False


def lark_message_is_from_bot(message_id: str) -> bool:
    try:
        resp = requests.get(
            f"https://open.larkoffice.com/open-apis/im/v1/messages/{message_id}",
            headers=lark_headers(),
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code >= 400:
            app.logger.warning("Failed to fetch message metadata id=%s status=%s", message_id, resp.status_code)
            return False
        body = resp.json()
        data = body.get("data") or {}
        candidates = []
        if isinstance(data.get("items"), list):
            candidates.extend(data.get("items"))
        if isinstance(data.get("message"), dict):
            candidates.append(data.get("message"))
        if isinstance(data, dict):
            candidates.append(data)

        for c in candidates:
            sender_type = str(c.get("sender_type") or ((c.get("sender") or {}).get("sender_type") or "")).lower()
            sender_open_id = (
                ((c.get("sender_id") or {}).get("open_id"))
                or (((c.get("sender") or {}).get("sender_id") or {}).get("open_id"))
                or (((c.get("sender") or {}).get("id") or {}).get("open_id"))
                or ""
            )
            if sender_type in {"bot", "app"}:
                return True
            if LARK_BOT_OPEN_ID and sender_open_id == LARK_BOT_OPEN_ID:
                return True
    except Exception:
        app.logger.exception("Failed checking whether message is from bot id=%s", message_id)
    return False


def sanitize_text(s: str) -> str:
    text = (s or "").strip()
    text = re.sub(r"@_user_\d+", "", text)
    text = re.sub(r"<at\\b[^>]*>.*?</at>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"@Sticksy\\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text_from_rich_content(obj) -> str:
    parts: list[str] = []

    def walk(x):
        if isinstance(x, dict):
            tag = str(x.get("tag") or "").lower()
            if tag == "text" and isinstance(x.get("text"), str):
                parts.append(x.get("text"))
            elif tag == "at":
                name = x.get("user_name") or x.get("name") or ""
                if isinstance(name, str) and name.strip():
                    parts.append(name.strip())
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for item in x:
                walk(item)

    walk(obj)
    return sanitize_text(" ".join(parts))


def extract_text_content(message_type: str, content_obj: dict) -> str:
    mt = (message_type or "").lower()
    if mt == "text":
        return sanitize_text(content_obj.get("text") or "")
    if mt in {"post", "interactive"}:
        return extract_text_from_rich_content(content_obj)
    return ""


def mention_open_id(mention: dict) -> str:
    mention_id = mention.get("id") or {}
    if isinstance(mention_id, dict):
        return (
            (mention_id.get("open_id") or "").strip()
            or (mention.get("open_id") or "").strip()
        )
    if isinstance(mention_id, str):
        return mention_id.strip() or (mention.get("open_id") or "").strip()
    return (mention.get("open_id") or "").strip()


def mention_display_name(mention: dict) -> str:
    return sanitize_text(
        (mention.get("name") or "").strip()
        or (mention.get("user_name") or "").strip()
        or (mention.get("display_name") or "").strip()
    )


def mention_id_candidates(mention: dict) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(value, id_type: str):
        if not isinstance(value, str):
            return
        clean_value = value.strip()
        clean_type = (id_type or "").strip().lower() or "open_id"
        if not clean_value:
            return
        item = (clean_value, clean_type)
        if item in seen:
            return
        seen.add(item)
        candidates.append(item)

    mention_id = mention.get("id") or {}
    if isinstance(mention_id, dict):
        add(mention_id.get("open_id"), "open_id")
        add(mention_id.get("user_id"), "user_id")
        add(mention_id.get("union_id"), "union_id")
        raw_id = mention_id.get("id")
        if isinstance(raw_id, str):
            add(raw_id, "open_id" if raw_id.strip().startswith("ou_") else "user_id")
    elif isinstance(mention_id, str):
        add(mention_id, "open_id" if mention_id.strip().startswith("ou_") else "user_id")

    add(mention.get("open_id"), "open_id")
    add(mention.get("user_id"), "user_id")
    add(mention.get("union_id"), "union_id")
    return candidates


def resolve_mention_identity(mention: dict) -> tuple[str, str]:
    display_name = mention_display_name(mention)
    for identifier, id_type in mention_id_candidates(mention):
        identity = fetch_lark_user_identity(identifier, user_id_type=id_type)
        open_id = (identity.get("open_id") or "").strip()
        resolved_name = (identity.get("display_name") or "").strip() or display_name
        if open_id:
            return open_id, resolved_name
    return "", display_name


def remember_non_bot_mentions(chat_id: str, mentions: list[dict]):
    saved = 0
    unresolved = 0
    for mention in mentions or []:
        if not isinstance(mention, dict):
            continue
        display_name = mention_display_name(mention)
        key_value = str(mention.get("key") or "").strip().lower()
        direct_open_id = mention_open_id(mention)
        if (LARK_BOT_OPEN_ID and direct_open_id == LARK_BOT_OPEN_ID) or display_name.lower() == "sticksy" or "sticksy" in key_value:
            continue
        open_id, resolved_name = resolve_mention_identity(mention)
        if not open_id:
            unresolved += 1
            continue
        if (LARK_BOT_OPEN_ID and open_id == LARK_BOT_OPEN_ID) or display_name.lower() == "sticksy" or "sticksy" in key_value:
            continue
        save_mentioned_identity(chat_id, open_id, resolved_name or display_name)
        saved += 1
    if saved:
        app.logger.info("Saved mentioned identities: chat_id=%s count=%s", chat_id, saved)
    if unresolved:
        app.logger.warning("Could not resolve mentioned identities: chat_id=%s count=%s", chat_id, unresolved)


def is_bot_mention(mention: dict) -> bool:
    open_id = mention_open_id(mention)
    if LARK_BOT_OPEN_ID:
        if open_id == LARK_BOT_OPEN_ID:
            return True
        name = mention_display_name(mention).lower()
        if name == "sticksy":
            return True
        key = str(mention.get("key") or "").strip().lower()
        if "sticksy" in key:
            return True
        return False
    # Fallback when bot open_id is not configured: treat any mention as bot mention.
    return True


def extract_bot_segments(raw_text: str, mentions: list[dict]) -> list[str]:
    if not raw_text:
        return []

    tokens = list(re.finditer(r"@_user_\d+", raw_text))
    bot_mentions = [m for m in mentions if is_bot_mention(m)]

    if not tokens:
        if bot_mentions:
            t = sanitize_text(raw_text)
            return [t] if t else []
        return []

    mention_by_key = {}
    for m in mentions:
        key = m.get("key")
        if isinstance(key, str) and key:
            mention_by_key[key] = m

    segments = []
    bot_token_count = 0
    for i, token in enumerate(tokens):
        mention = mention_by_key.get(token.group(0))
        if not mention and i < len(mentions):
            mention = mentions[i]
        if not mention or not is_bot_mention(mention):
            continue
        bot_token_count += 1

        start = token.end()
        end = tokens[i + 1].start() if i + 1 < len(tokens) else len(raw_text)
        seg = sanitize_text(raw_text[start:end])
        if seg:
            segments.append(seg)

    if not segments and bot_token_count == 1:
        # Support patterns like "Hi @Sticksy" where user text appears before the mention.
        cleaned = sanitize_text(re.sub(r"@_user_\d+", "", raw_text))
        if cleaned:
            segments.append(cleaned)

    return segments


def looks_like_summary_request(text: str) -> bool:
    t = text.lower()
    keywords = ["summarize", "summary", "sum up", "recap", "recap it", "总结", "總結", "概括", "回顾", "回顧"]
    return any(k in t for k in keywords)


@dataclass
class TimeWindow:
    start_ts: int
    end_ts: int
    label: str


def parse_time_window(query: str, user_tz: str) -> TimeWindow:
    tz = ZoneInfo(user_tz)
    now_local = datetime.now(tz)
    q = query.lower()

    # last Xh / Xm / Xd
    m = re.search(r"last\s+(\d+)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes|d|day|days)\b", q)
    if not m:
        m = re.search(r"last\s*(\d+)\s*(h|m|d)\b", q)
    if m:
        amount = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("h"):
            delta = timedelta(hours=amount)
            label = f"last {amount}h"
        elif unit.startswith("m"):
            delta = timedelta(minutes=amount)
            label = f"last {amount}m"
        else:
            delta = timedelta(days=amount)
            label = f"last {amount}d"
        start_local = now_local - delta
        return TimeWindow(int(start_local.timestamp()), int(now_local.timestamp()), label)

    if "yesterday" in q or "昨天" in q:
        day = (now_local - timedelta(days=1)).date()
        start = datetime(day.year, day.month, day.day, tzinfo=tz)
        end = start + timedelta(days=1)
        return TimeWindow(int(start.timestamp()), int(end.timestamp()), "yesterday")

    # default: today
    day = now_local.date()
    start = datetime(day.year, day.month, day.day, tzinfo=tz)
    end = now_local
    return TimeWindow(int(start.timestamp()), int(end.timestamp()), "today")


def get_user_profile(sender_open_id: str) -> dict:
    if not sender_open_id:
        return {"tz_name": "UTC", "display_name": ""}

    row = db_query_one(
        "SELECT tz_name, display_name, cached_at_ts FROM user_profile_cache WHERE open_id = :open_id",
        {"open_id": sender_open_id},
    )
    if row and now_ts() - int(row["cached_at_ts"]) < 86400:
        return {"tz_name": row["tz_name"] or "UTC", "display_name": row["display_name"] or ""}

    tz_name = "UTC"
    display_name = ""
    try:
        resp = requests.get(
            f"https://open.larkoffice.com/open-apis/contact/v3/users/{sender_open_id}",
            params={"user_id_type": "open_id"},
            headers=lark_headers(),
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code < 400:
            body = resp.json()
            user = ((body.get("data") or {}).get("user") or {})
            tz = user.get("timezone")
            if tz:
                tz_name = tz
            display_name = (user.get("name") or user.get("en_name") or "").strip()
    except Exception:
        app.logger.exception("Failed to fetch user profile for open_id=%s", sender_open_id)

    try:
        _ = ZoneInfo(tz_name)
    except Exception:
        tz_name = "UTC"

    if IS_POSTGRES:
        db_execute(
            """
            INSERT INTO user_profile_cache(open_id, tz_name, display_name, cached_at_ts)
            VALUES(:open_id, :tz_name, :display_name, :cached_at_ts)
            ON CONFLICT (open_id) DO UPDATE
            SET tz_name = EXCLUDED.tz_name, display_name = EXCLUDED.display_name, cached_at_ts = EXCLUDED.cached_at_ts
            """,
            {
                "open_id": sender_open_id,
                "tz_name": tz_name,
                "display_name": display_name,
                "cached_at_ts": now_ts(),
            },
        )
    else:
        db_execute(
            "INSERT OR REPLACE INTO user_profile_cache(open_id, tz_name, display_name, cached_at_ts) VALUES(:open_id, :tz_name, :display_name, :cached_at_ts)",
            {
                "open_id": sender_open_id,
                "tz_name": tz_name,
                "display_name": display_name,
                "cached_at_ts": now_ts(),
            },
        )

    return {"tz_name": tz_name, "display_name": display_name}


def fetch_lark_user_identity(user_id: str, user_id_type: str = "open_id") -> dict:
    clean_id = (user_id or "").strip()
    clean_type = (user_id_type or "open_id").strip().lower()
    if not clean_id:
        return {"open_id": "", "display_name": ""}
    if clean_type not in {"open_id", "user_id", "union_id"}:
        clean_type = "open_id"

    if clean_type == "open_id":
        profile = get_user_profile(clean_id)
        return {
            "open_id": clean_id,
            "display_name": (profile.get("display_name") or "").strip(),
        }

    try:
        resp = requests.get(
            f"https://open.larkoffice.com/open-apis/contact/v3/users/{quote(clean_id, safe='')}",
            params={"user_id_type": clean_type},
            headers=lark_headers(),
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code >= 400:
            app.logger.warning("User identity lookup failed type=%s status=%s", clean_type, resp.status_code)
            return {"open_id": "", "display_name": ""}
        body = resp.json()
        if body.get("code", 0) != 0:
            app.logger.warning(
                "User identity lookup api error type=%s code=%s msg=%s",
                clean_type,
                body.get("code"),
                body.get("msg"),
            )
            return {"open_id": "", "display_name": ""}
        user = ((body.get("data") or {}).get("user") or {})
        return {
            "open_id": (user.get("open_id") or "").strip(),
            "display_name": ((user.get("name") or user.get("en_name") or "")).strip(),
        }
    except Exception:
        app.logger.exception("User identity lookup exception type=%s", clean_type)
        return {"open_id": "", "display_name": ""}


def get_user_timezone(sender_open_id: str) -> str:
    profile = get_user_profile(sender_open_id)
    return profile.get("tz_name") or "UTC"


def load_messages_for_window(chat_id: str, start_ts: int, end_ts: int, root_id: str | None) -> list[dict]:
    params = {"chat_id": chat_id, "start_ts": start_ts, "end_ts": end_ts, "max_rows": MAX_SUMMARY_MESSAGES}
    sql = (
        "SELECT sender_name, sender_open_id, message_type, text_content, image_key, created_at_ts, root_id "
        "FROM messages WHERE chat_id = :chat_id AND created_at_ts >= :start_ts AND created_at_ts < :end_ts"
    )
    if root_id:
        sql += " AND root_id = :root_id"
        params["root_id"] = root_id
    sql += " ORDER BY created_at_ts ASC LIMIT :max_rows"
    return db_query_all(sql, params)


def create_summary(chat_id: str, query: str, asker_name: str, language_hint: str, rows: list[dict], window_label: str) -> tuple[str, list[str]]:
    message_lines = []
    image_pool = []
    speaker_cache: dict[str, str] = {}
    for row in rows:
        ts = datetime.fromtimestamp(int(row["created_at_ts"]), tz=timezone.utc).strftime("%H:%M")
        speaker = (row.get("sender_name") or "").strip()
        sender_open_id = (row.get("sender_open_id") or "").strip()
        if (not speaker or speaker.lower() == "unknown") and sender_open_id:
            if sender_open_id not in speaker_cache:
                profile = get_user_profile(sender_open_id)
                speaker_cache[sender_open_id] = (profile.get("display_name") or "").strip()
            speaker = speaker_cache.get(sender_open_id) or speaker
        if not speaker:
            speaker = "Unknown"
        if row["message_type"] == "text":
            text = (row["text_content"] or "").strip()
            if text:
                message_lines.append(f"[{ts}] {speaker}: {text}")
        elif row["message_type"] == "image" and row["image_key"]:
            image_key = row["image_key"]
            message_lines.append(f"[{ts}] {speaker}: [shared image: {image_key}]")
            image_pool.append(image_key)

    user_block = {
        "request": query,
        "asker_name": asker_name,
        "window_label": window_label,
        "language_hint": language_hint,
        "learned_terms": load_learned_terms(chat_id),
        "messages": message_lines,
        "image_pool": image_pool,
    }

    config = types.GenerateContentConfig(
        system_instruction=SUMMARY_SYSTEM_PROMPT,
        response_mime_type="application/json",
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=json.dumps(user_block, ensure_ascii=False),
        config=config,
    )
    raw = (response.text or "").strip()
    parsed = json.loads(raw)

    intro = (parsed.get("intro") or "").strip()
    topics = parsed.get("topics") or []
    bullets = []
    for t in topics:
        summary = (t.get("summary") or "").strip()
        if not summary:
            continue
        bullets.append(f"- {summary}")

    if window_label == "today":
        first_line = "Summary of today's conversation:"
    elif window_label == "yesterday":
        first_line = "Summary of yesterday's conversation:"
    else:
        first_line = f"Summary of {window_label} conversation:"

    summary_parts = [first_line]
    if bullets:
        summary_parts.append("\n".join(bullets))
    summary_text = "\n".join(summary_parts)

    image_candidates = parsed.get("important_image_keys") or []
    selected = []
    valid = set(image_pool)
    for key in image_candidates:
        if key in valid and key not in selected:
            selected.append(key)
        if len(selected) >= MAX_SUMMARY_IMAGES:
            break

    return summary_text, selected


def extract_minutes_url(text: str) -> str:
    match = re.search(r"https?://[^\s]+/minutes/[A-Za-z0-9]+[^\s]*", text or "", re.IGNORECASE)
    return match.group(0).rstrip(").,]") if match else ""


def extract_minutes_token(minutes_url: str) -> str:
    match = re.search(r"/minutes/([A-Za-z0-9]+)", minutes_url or "", re.IGNORECASE)
    return match.group(1) if match else ""


def load_cached_meeting_summary(minute_token: str, mode: str) -> dict | None:
    clean_token = (minute_token or "").strip()
    clean_mode = (mode or "").strip() or "summary"
    if not clean_token:
        return None
    return db_query_one(
        """
        SELECT minute_token, mode, reply_text, title, source_chat_id, created_at_ts
        FROM meeting_summary_cache
        WHERE minute_token = :minute_token AND mode = :mode
        LIMIT 1
        """,
        {"minute_token": clean_token, "mode": clean_mode},
    )


def save_cached_meeting_summary(minute_token: str, mode: str, reply_text: str, title: str, source_chat_id: str):
    clean_token = (minute_token or "").strip()
    clean_mode = (mode or "").strip() or "summary"
    clean_reply = (reply_text or "").strip()
    clean_title = (title or "").strip() or "Meeting"
    if not clean_token or not clean_reply:
        return

    params = {
        "minute_token": clean_token,
        "mode": clean_mode,
        "reply_text": clean_reply,
        "title": clean_title,
        "source_chat_id": (source_chat_id or "").strip(),
        "created_at_ts": now_ts(),
    }
    if IS_POSTGRES:
        db_execute(
            """
            INSERT INTO meeting_summary_cache(minute_token, mode, reply_text, title, source_chat_id, created_at_ts)
            VALUES(:minute_token, :mode, :reply_text, :title, :source_chat_id, :created_at_ts)
            ON CONFLICT (minute_token, mode) DO UPDATE
            SET reply_text = EXCLUDED.reply_text,
                title = EXCLUDED.title,
                source_chat_id = EXCLUDED.source_chat_id,
                created_at_ts = EXCLUDED.created_at_ts
            """,
            params,
        )
    else:
        db_execute(
            """
            INSERT OR REPLACE INTO meeting_summary_cache(minute_token, mode, reply_text, title, source_chat_id, created_at_ts)
            VALUES(:minute_token, :mode, :reply_text, :title, :source_chat_id, :created_at_ts)
            """,
            params,
        )


def meeting_request_mode(text: str) -> str:
    lowered = (text or "").lower()
    has_url = bool(extract_minutes_url(text))
    has_meeting_hint = any(word in lowered for word in ["meeting", "minutes", "transcript", "会议", "會議", "纪要", "紀要"])
    next_steps_hint = any(phrase in lowered for phrase in ["next steps", "action items", "todos", "todo", "下一步", "待办", "待辦"])
    summary_hint = looks_like_summary_request(text)

    if next_steps_hint and (has_url or has_meeting_hint):
        return "next_steps"
    if has_url:
        return "summary"
    if summary_hint and has_meeting_hint:
        return "summary"
    return ""


def prefers_requester_owned_recent_meeting(text: str) -> bool:
    lowered = (text or "").lower()
    explicit_phrases = [
        "my previous meeting",
        "my last meeting",
        "my latest meeting",
        "my recent meeting",
        "meeting i owned",
        "meeting i was the owner of",
        "meeting i hosted",
        "meeting i organized",
        "meeting i organised",
        "meeting i led",
    ]
    implicit_recent_phrases = [
        "previous meeting",
        "last meeting",
        "latest meeting",
        "recent meeting",
    ]
    return any(phrase in lowered for phrase in explicit_phrases + implicit_recent_phrases)


def recent_meeting_urls(chat_id: str, scan_limit: int = 80, unique_limit: int = 12) -> list[str]:
    rows = db_query_all(
        """
        SELECT text_content
        FROM messages
        WHERE chat_id = :chat_id
          AND text_content IS NOT NULL
          AND text_content <> ''
        ORDER BY created_at_ts DESC
        LIMIT :limit_rows
        """,
        {"chat_id": chat_id, "limit_rows": scan_limit},
    )

    urls: list[str] = []
    seen: set[str] = set()
    for row in rows:
        url = extract_minutes_url(row.get("text_content") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
        if len(urls) >= unique_limit:
            break
    return urls


def find_last_meeting_url(chat_id: str, limit: int = 80) -> str:
    urls = recent_meeting_urls(chat_id, scan_limit=limit, unique_limit=1)
    return urls[0] if urls else ""


def lark_auth_headers() -> dict:
    return {"Authorization": f"Bearer {get_tenant_access_token()}"}


def fetch_lark_minute_meta(minute_token: str) -> dict:
    if not minute_token:
        return {}
    resp = requests.get(
        f"https://open.larkoffice.com/open-apis/minutes/v1/minutes/{minute_token}",
        headers=lark_auth_headers(),
        timeout=HTTP_TIMEOUT,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"minute meta fetch failed status={resp.status_code}: {resp.text[:300]}")
    body = resp.json()
    if body.get("code", 0) != 0:
        raise RuntimeError(f"minute meta api error code={body.get('code')} msg={body.get('msg')}")
    return (body.get("data") or {})


def fetch_lark_minute_statistics(minute_token: str) -> dict:
    if not minute_token:
        return {}

    urls = [
        f"https://open.larkoffice.com/open-apis/minutes/v1/minutes/{minute_token}/statistics",
        f"https://open.larkoffice.com/open-apis/minutes/v1/{minute_token}/statistics",
    ]

    for idx, url in enumerate(urls):
        try:
            resp = requests.get(
                url,
                headers=lark_auth_headers(),
                timeout=HTTP_TIMEOUT,
            )
            if resp.status_code == 404 and idx == 0:
                continue
            if resp.status_code >= 400:
                app.logger.warning(
                    "Minutes statistics unavailable: token=%s status=%s body=%s",
                    minute_token,
                    resp.status_code,
                    resp.text[:300],
                )
                return {}
            body = resp.json()
            if body.get("code", 0) != 0:
                app.logger.warning(
                    "Minutes statistics api error: token=%s code=%s msg=%s",
                    minute_token,
                    body.get("code"),
                    body.get("msg"),
                )
                return {}
            data = body.get("data") or {}
            try:
                top_keys = sorted([str(k) for k in data.keys()]) if isinstance(data, dict) else []
                sample = json.dumps(data, ensure_ascii=False)[:1200]
                app.logger.info(
                    "Minutes statistics payload: token=%s keys=%s sample=%s",
                    minute_token,
                    top_keys,
                    sample,
                )
            except Exception:
                app.logger.exception("Failed to summarize minutes statistics payload: token=%s", minute_token)
            return data
        except Exception:
            app.logger.exception("Minutes statistics request failed: token=%s", minute_token)
            return {}

    return {}


def extract_transcript_text_from_obj(obj) -> str:
    text_parts: list[str] = []

    def walk(node, parent_key: str = ""):
        if isinstance(node, dict):
            for key, value in node.items():
                key_l = str(key).lower()
                if isinstance(value, str):
                    val = value.strip()
                    if not val:
                        continue
                    if val.startswith("http://") or val.startswith("https://"):
                        continue
                    if key_l in {
                        "text", "content", "paragraph_text", "sentence_text",
                        "transcript", "topic", "summary", "speaker", "speaker_name",
                        "email", "speaker_email", "user_email", "email_address",
                    }:
                        text_parts.append(val)
                else:
                    walk(value, key_l)
        elif isinstance(node, list):
            for item in node:
                walk(item, parent_key)

    walk(obj)
    merged = "\n".join(text_parts)
    merged = re.sub(r"\n{3,}", "\n\n", merged).strip()
    if len(merged) > MAX_MEETING_TRANSCRIPT_CHARS:
        merged = merged[:MAX_MEETING_TRANSCRIPT_CHARS]
    return merged


def first_nested_string(node, keys: set[str], max_depth: int = 1) -> str:
    if max_depth < 0:
        return ""
    if isinstance(node, dict):
        for key, value in node.items():
            key_l = str(key).lower()
            if key_l in keys and isinstance(value, str) and value.strip():
                return value.strip()
        if max_depth > 0:
            for value in node.values():
                found = first_nested_string(value, keys, max_depth=max_depth - 1)
                if found:
                    return found
    elif isinstance(node, list) and max_depth > 0:
        for item in node:
            found = first_nested_string(item, keys, max_depth=max_depth - 1)
            if found:
                return found
    return ""


def first_nested_id(node, max_depth: int = 1) -> str:
    if max_depth < 0:
        return ""
    if isinstance(node, dict):
        for key, value in node.items():
            key_l = str(key).lower()
            if key_l in {"open_id", "user_id", "id"} and isinstance(value, str) and value.strip():
                return value.strip()
            if key_l == "member_id":
                if isinstance(value, str) and value.strip():
                    return value.strip()
                if isinstance(value, dict):
                    nested_id = (
                        (value.get("open_id") or "").strip()
                        or (value.get("user_id") or "").strip()
                        or (value.get("id") or "").strip()
                    )
                    if nested_id:
                        return nested_id
        if max_depth > 0:
            for value in node.values():
                found = first_nested_id(value, max_depth=max_depth - 1)
                if found:
                    return found
    elif isinstance(node, list) and max_depth > 0:
        for item in node:
            found = first_nested_id(item, max_depth=max_depth - 1)
            if found:
                return found
    return ""


def decode_response_text(resp: requests.Response) -> str:
    payload = resp.content or b""
    if not payload:
        return ""
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return payload.decode(encoding).strip()
        except Exception:
            pass
    apparent = getattr(resp, "apparent_encoding", None)
    if apparent:
        try:
            return payload.decode(apparent, errors="replace").strip()
        except Exception:
            pass
    try:
        return resp.text.strip()
    except Exception:
        return payload.decode("latin-1", errors="replace").strip()


def extract_speaker_directory_from_obj(obj) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()

    name_keys = {"speaker", "speaker_name", "name", "user_name", "display_name"}
    email_keys = {"email", "speaker_email", "user_email", "email_address"}

    def maybe_add(node):
        if not isinstance(node, dict):
            return
        speaker_name = sanitize_text(first_nested_string(node, name_keys, max_depth=1))
        speaker_email = first_nested_string(node, email_keys, max_depth=1).lower()
        if speaker_name and speaker_email:
            pair_key = (normalize_person_name(speaker_name), speaker_email)
            if pair_key[0] and pair_key not in seen:
                seen.add(pair_key)
                out.append({"name": speaker_name, "email": speaker_email})

    def walk(node):
        if isinstance(node, dict):
            maybe_add(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)
    return out


def extract_participant_directory_from_obj(obj, allow_name_only: bool = False) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    name_keys = {
        "speaker", "speaker_name", "name", "user_name", "display_name",
        "participant_name", "attendee_name", "nick_name", "nickname", "full_name", "en_name",
    }
    email_keys = {"email", "speaker_email", "user_email", "email_address"}
    participant_hints = ("participant", "attendee", "speaker", "member", "people", "user", "owner", "invitee")

    def maybe_add(node, parent_key: str = "", context_hint: bool = False):
        if not isinstance(node, dict):
            return

        participant_name = sanitize_text(first_nested_string(node, name_keys, max_depth=2))
        participant_email = first_nested_string(node, email_keys, max_depth=2).lower()
        participant_open_id = first_nested_id(node, max_depth=2)

        if not participant_name:
            return
        if not participant_email and not participant_open_id:
            if not allow_name_only:
                return
            parent_hint = (parent_key or "").lower()
            node_keys = {str(k).lower() for k in node.keys()}
            has_context_hint = context_hint or any(hint in parent_hint for hint in participant_hints)
            has_node_hint = any(hint in key for key in node_keys for hint in participant_hints)
            has_explicit_name_hint = bool(node_keys & {"participant_name", "attendee_name", "speaker_name", "user_name", "display_name"})
            if not (has_context_hint or has_node_hint or has_explicit_name_hint):
                return

        item_key = (normalize_person_name(participant_name), participant_email, participant_open_id)
        if item_key[0] and item_key not in seen:
            seen.add(item_key)
            out.append(
                {
                    "name": participant_name,
                    "email": participant_email,
                    "open_id": participant_open_id,
                }
            )

    def walk(node, parent_key: str = "", context_hint: bool = False):
        if isinstance(node, dict):
            node_keys = {str(k).lower() for k in node.keys()}
            node_hint = context_hint or any(hint in key for key in node_keys for hint in participant_hints)
            maybe_add(node, parent_key=parent_key, context_hint=node_hint)
            for key, value in node.items():
                child_key = str(key).lower()
                child_hint = node_hint or any(hint in child_key for hint in participant_hints)
                walk(value, parent_key=child_key, context_hint=child_hint)
        elif isinstance(node, list):
            for item in node:
                walk(item, parent_key=parent_key, context_hint=context_hint)

    walk(obj)
    return out


def merge_people_directories(*directories: list[dict] | None) -> list[dict]:
    merged: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for directory in directories:
        for item in directory or []:
            name = (item.get("name") or "").strip()
            email = (item.get("email") or "").strip().lower()
            open_id = (
                (item.get("open_id") or "").strip()
                or (item.get("user_id") or "").strip()
                or (item.get("id") or "").strip()
            )
            key = (normalize_person_name(name), email, open_id)
            if not key[0] or key in seen:
                continue
            seen.add(key)
            merged.append({"name": name, "email": email, "open_id": open_id})

    return merged


def extract_people_directory_from_text_blob(blob: str) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    if not blob:
        return out

    variants = []
    for candidate in [
        blob,
        html.unescape(blob),
        blob.replace('\\"', '"').replace("\\/", "/"),
    ]:
        normalized = (candidate or "").strip()
        if normalized and normalized not in variants:
            variants.append(normalized)

    name_key = r'(?:name|display_name|user_name|participant_name|attendee_name|speaker_name)'
    email_key = r'(?:email|user_email|speaker_email|email_address)'
    id_key = r'(?:open_id|user_id|member_id|id)'

    patterns = [
        re.compile(
            rf'"{name_key}"\s*:\s*"([^"]{{1,120}})".{{0,400}}?"{email_key}"\s*:\s*"([^"]+@[^"]+)"(?:.{{0,200}}?"{id_key}"\s*:\s*"([^"]+)")?',
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            rf'"{email_key}"\s*:\s*"([^"]+@[^"]+)".{{0,400}}?"{name_key}"\s*:\s*"([^"]{{1,120}})"(?:.{{0,200}}?"{id_key}"\s*:\s*"([^"]+)")?',
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            rf'"{name_key}"\s*:\s*"([^"]{{1,120}})".{{0,300}}?"{id_key}"\s*:\s*"([^"]+)"',
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            rf'"{id_key}"\s*:\s*"([^"]+)".{{0,300}}?"{name_key}"\s*:\s*"([^"]{{1,120}})"',
            re.IGNORECASE | re.DOTALL,
        ),
    ]

    for text_blob in variants:
        for match in patterns[0].finditer(text_blob):
            name = sanitize_text(match.group(1) or "")
            email_value = (match.group(2) or "").strip().lower()
            open_id = (match.group(3) or "").strip()
            key = (normalize_person_name(name), email_value, open_id)
            if key[0] and (email_value or open_id) and key not in seen:
                seen.add(key)
                out.append({"name": name, "email": email_value, "open_id": open_id})

        for match in patterns[1].finditer(text_blob):
            email_value = (match.group(1) or "").strip().lower()
            name = sanitize_text(match.group(2) or "")
            open_id = (match.group(3) or "").strip()
            key = (normalize_person_name(name), email_value, open_id)
            if key[0] and (email_value or open_id) and key not in seen:
                seen.add(key)
                out.append({"name": name, "email": email_value, "open_id": open_id})

        for match in patterns[2].finditer(text_blob):
            name = sanitize_text(match.group(1) or "")
            open_id = (match.group(2) or "").strip()
            key = (normalize_person_name(name), "", open_id)
            if key[0] and open_id and key not in seen:
                seen.add(key)
                out.append({"name": name, "email": "", "open_id": open_id})

        for match in patterns[3].finditer(text_blob):
            open_id = (match.group(1) or "").strip()
            name = sanitize_text(match.group(2) or "")
            key = (normalize_person_name(name), "", open_id)
            if key[0] and open_id and key not in seen:
                seen.add(key)
                out.append({"name": name, "email": "", "open_id": open_id})

    return out


def extract_name_like_values_from_obj(obj, limit: int = 30) -> list[str]:
    if obj is None:
        return []

    name_keys = {
        "speaker", "speaker_name", "name", "user_name", "display_name",
        "participant_name", "attendee_name", "nick_name", "nickname", "full_name", "en_name",
    }
    out: list[str] = []
    seen: set[str] = set()

    def maybe_add(value):
        if not isinstance(value, str):
            return
        cleaned = sanitize_text(value)
        if not cleaned:
            return
        if cleaned.startswith(("http://", "https://")):
            return
        if len(cleaned) > 80:
            return
        if not re.search(r"[A-Za-z\u4e00-\u9fff]", cleaned):
            return
        key = normalize_person_name(cleaned)
        if not key or key in seen:
            return
        seen.add(key)
        out.append(cleaned)

    def walk(node):
        if len(out) >= limit:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                key_l = str(key).lower()
                if key_l in name_keys:
                    maybe_add(value)
                    if len(out) >= limit:
                        return
                walk(value)
                if len(out) >= limit:
                    return
        elif isinstance(node, list):
            for item in node:
                walk(item)
                if len(out) >= limit:
                    return

    walk(obj)
    return out


def extract_name_like_values_from_public_html(page_text: str, limit: int = 30) -> list[str]:
    if not page_text:
        return []

    out: list[str] = []
    seen: set[str] = set()
    for blob in extract_json_like_blobs_from_html(page_text):
        parsed = parse_json_like_blob(blob)
        if parsed is None:
            continue
        for name in extract_name_like_values_from_obj(parsed, limit=limit):
            key = normalize_person_name(name)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(name)
            if len(out) >= limit:
                return out
    return out


def parse_json_like_blob(blob: str):
    if not blob:
        return None

    candidates = []
    stripped = blob.strip()
    if stripped:
        candidates.append(stripped)
        candidates.append(html.unescape(stripped))

    parse_match = re.search(r'JSON\.parse\(\s*"((?:\\.|[^"])*)"\s*\)', stripped, re.DOTALL)
    if parse_match:
        try:
            decoded = json.loads(f'"{parse_match.group(1)}"')
            if decoded:
                candidates.append(decoded)
        except Exception:
            pass

    for candidate in candidates:
        text_candidate = candidate.strip() if isinstance(candidate, str) else candidate
        if isinstance(text_candidate, str):
            if not text_candidate:
                continue
            try:
                return json.loads(text_candidate)
            except Exception:
                pass
        else:
            try:
                return json.loads(text_candidate)
            except Exception:
                continue

    return None


def extract_json_like_blobs_from_html(page_text: str) -> list[str]:
    if not page_text:
        return []

    blobs: list[str] = []
    script_patterns = [
        re.compile(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.IGNORECASE | re.DOTALL),
        re.compile(r'<script[^>]*>\s*window\.__INITIAL_STATE__\s*=\s*(.*?);\s*</script>', re.IGNORECASE | re.DOTALL),
        re.compile(r'<script[^>]*>\s*window\.__NUXT__\s*=\s*(.*?);\s*</script>', re.IGNORECASE | re.DOTALL),
        re.compile(r'window\.__INITIAL_STATE__\s*=\s*(\{.*?\})\s*;', re.IGNORECASE | re.DOTALL),
        re.compile(r'window\.__NEXT_DATA__\s*=\s*(\{.*?\})\s*;', re.IGNORECASE | re.DOTALL),
    ]

    seen_blob = set()
    for pattern in script_patterns:
        for match in pattern.finditer(page_text):
            blob = (match.group(1) or "").strip()
            if not blob or blob in seen_blob:
                continue
            seen_blob.add(blob)
            blobs.append(blob)

    return blobs


def extract_people_directory_from_public_html(page_text: str) -> list[dict]:
    if not page_text:
        return []

    blobs = extract_json_like_blobs_from_html(page_text)
    collected: list[dict] = []
    for blob in blobs:
        parsed = parse_json_like_blob(blob)
        if parsed is None:
            continue
        collected = merge_people_directories(
            collected,
            extract_participant_directory_from_obj(parsed, allow_name_only=True),
            extract_speaker_directory_from_obj(parsed),
        )

    if collected:
        return collected

    return extract_people_directory_from_text_blob(page_text)


def extract_transcript_text_from_public_html(page_text: str) -> str:
    if not page_text:
        return ""

    best = ""
    for blob in extract_json_like_blobs_from_html(page_text):
        parsed = parse_json_like_blob(blob)
        if parsed is None:
            continue
        extracted = extract_transcript_text_from_obj(parsed)
        if not extracted:
            continue
        if len(extracted) > len(best):
            best = extracted

    return best[:MAX_MEETING_TRANSCRIPT_CHARS] if best else ""


def fetch_public_minute_page_text(minute_url: str) -> str:
    if not minute_url:
        return ""

    try:
        resp = requests.get(
            minute_url,
            headers={"User-Agent": "SticksyBot/1.0"},
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code >= 400:
            app.logger.warning(
                "Public minute page unavailable: url=%s status=%s body=%s",
                minute_url,
                resp.status_code,
                resp.text[:300],
            )
            return ""

        page_text = decode_response_text(resp)
        if not page_text:
            app.logger.warning("Public minute page empty: url=%s", minute_url)
            return ""
        return page_text
    except Exception:
        app.logger.exception("Failed fetching public minute page: url=%s", minute_url)
        return ""


def fetch_public_minute_page_people(minute_url: str, page_text: str | None = None) -> list[dict]:
    if not minute_url:
        return []

    resolved_text = page_text or fetch_public_minute_page_text(minute_url)
    if not resolved_text:
        return []

    people = extract_people_directory_from_public_html(resolved_text)
    if people:
        app.logger.info("Public minute page people extracted: url=%s count=%s", minute_url, len(people))
    else:
        app.logger.warning("Public minute page yielded no people: url=%s", minute_url)
    return people


def fetch_lark_minute_media(minute_token: str) -> tuple[bytes, str] | tuple[None, None]:
    if not minute_token:
        return (None, None)

    resp = requests.get(
        f"https://open.larkoffice.com/open-apis/minutes/v1/minutes/{minute_token}/media",
        headers=lark_auth_headers(),
        timeout=HTTP_TIMEOUT,
        allow_redirects=False,
    )
    app.logger.info(
        "Minutes media request: token=%s status=%s content_type=%s",
        minute_token,
        resp.status_code,
        resp.headers.get("Content-Type"),
    )

    if resp.status_code in {301, 302, 303, 307, 308} and resp.headers.get("Location"):
        app.logger.info("Minutes media redirected: token=%s", minute_token)
        media_resp = requests.get(resp.headers["Location"], timeout=HTTP_TIMEOUT)
    elif resp.status_code < 400 and "application/json" in (resp.headers.get("Content-Type") or "").lower():
        body = resp.json()
        if body.get("code", 0) != 0:
            app.logger.warning(
                "Minutes media api error: token=%s code=%s msg=%s",
                minute_token,
                body.get("code"),
                body.get("msg"),
            )
            return (None, None)
        data = body.get("data") or {}
        download_url = (
            data.get("download_url")
            or data.get("url")
            or ((data.get("media") or {}).get("download_url"))
            or ((data.get("media") or {}).get("url"))
        )
        if not download_url:
            app.logger.warning("Minutes media missing download URL: token=%s body=%s", minute_token, json.dumps(body)[:300])
            return (None, None)
        media_resp = requests.get(download_url, timeout=HTTP_TIMEOUT)
    elif resp.status_code < 400:
        media_resp = resp
    else:
        app.logger.warning("Minutes media denied/unavailable: token=%s status=%s body=%s", minute_token, resp.status_code, resp.text[:300])
        return (None, None)

    if media_resp.status_code >= 400:
        app.logger.warning("Minutes media download failed: token=%s status=%s", minute_token, media_resp.status_code)
        return (None, None)
    content = media_resp.content
    if not content or len(content) > MAX_MEETING_MEDIA_BYTES:
        app.logger.warning(
            "Minutes media empty/too large: token=%s size=%s max=%s",
            minute_token,
            len(content or b""),
            MAX_MEETING_MEDIA_BYTES,
        )
        return (None, None)
    mime_type = (media_resp.headers.get("Content-Type") or "audio/mp4").split(";")[0].strip() or "audio/mp4"
    app.logger.info("Minutes media ready: token=%s size=%s mime=%s", minute_token, len(content), mime_type)
    return (content, mime_type)


def fetch_lark_minute_transcript(minute_token: str) -> tuple[str, dict | None]:
    if not minute_token:
        return ("", None)

    resp = requests.get(
        f"https://open.larkoffice.com/open-apis/minutes/v1/minutes/{minute_token}/transcript",
        headers=lark_auth_headers(),
        timeout=HTTP_TIMEOUT,
        allow_redirects=False,
    )
    app.logger.info(
        "Minutes transcript request: token=%s status=%s content_type=%s",
        minute_token,
        resp.status_code,
        resp.headers.get("Content-Type"),
    )

    transcript_resp = None
    if resp.status_code in {301, 302, 303, 307, 308} and resp.headers.get("Location"):
        app.logger.info("Minutes transcript redirected: token=%s", minute_token)
        transcript_resp = requests.get(resp.headers["Location"], timeout=HTTP_TIMEOUT)
    elif resp.status_code < 400:
        transcript_resp = resp
    else:
        app.logger.warning("Minutes transcript denied/unavailable: token=%s status=%s body=%s", minute_token, resp.status_code, resp.text[:300])
        return ("", None)

    if transcript_resp.status_code >= 400:
        app.logger.warning("Minutes transcript download failed: token=%s status=%s", minute_token, transcript_resp.status_code)
        return ("", None)

    content_type = (transcript_resp.headers.get("Content-Type") or "").lower()
    if "application/json" not in content_type:
        text_payload = decode_response_text(transcript_resp)
        if len(text_payload) > MAX_MEETING_TRANSCRIPT_CHARS:
            text_payload = text_payload[:MAX_MEETING_TRANSCRIPT_CHARS]
        app.logger.info("Minutes transcript plain text length=%s token=%s", len(text_payload), minute_token)
        return (text_payload, None)

    try:
        body = transcript_resp.json()
    except Exception:
        app.logger.exception("Minutes transcript JSON parse failed: token=%s", minute_token)
        return ("", None)

    if body.get("code", 0) not in {0, None}:
        app.logger.warning(
            "Minutes transcript api error: token=%s code=%s msg=%s",
            minute_token,
            body.get("code"),
            body.get("msg"),
        )
        return ("", None)

    data = body.get("data") or {}
    speaker_directory = extract_speaker_directory_from_obj(data)
    speaker_prefix = "\n".join(
        f'{(item.get("name") or "").strip()} <{(item.get("email") or "").strip()}>'
        for item in speaker_directory
        if (item.get("name") or "").strip() and (item.get("email") or "").strip()
    ).strip()

    direct_text = (
        (data.get("transcript") if isinstance(data.get("transcript"), str) else "")
        or (data.get("content") if isinstance(data.get("content"), str) else "")
        or (data.get("text") if isinstance(data.get("text"), str) else "")
    ).strip()
    if direct_text:
        if speaker_prefix:
            direct_text = f"{speaker_prefix}\n{direct_text}".strip()
        app.logger.info("Minutes transcript direct text length=%s token=%s", len(direct_text), minute_token)
        return (direct_text[:MAX_MEETING_TRANSCRIPT_CHARS], data)

    download_url = (
        data.get("download_url")
        or data.get("url")
        or ((data.get("transcript") or {}).get("download_url") if isinstance(data.get("transcript"), dict) else "")
        or ((data.get("transcript") or {}).get("url") if isinstance(data.get("transcript"), dict) else "")
    )
    if isinstance(download_url, str) and download_url:
        try:
            download_resp = requests.get(download_url, timeout=HTTP_TIMEOUT)
            if download_resp.status_code < 400:
                text_payload = decode_response_text(download_resp)
                if speaker_prefix:
                    text_payload = f"{speaker_prefix}\n{text_payload}".strip()
                if len(text_payload) > MAX_MEETING_TRANSCRIPT_CHARS:
                    text_payload = text_payload[:MAX_MEETING_TRANSCRIPT_CHARS]
                app.logger.info("Minutes transcript export downloaded length=%s token=%s", len(text_payload), minute_token)
                return (text_payload, data)
            app.logger.warning("Minutes transcript export download failed: token=%s status=%s", minute_token, download_resp.status_code)
        except Exception:
            app.logger.exception("Failed to download minute transcript export")

    extracted = extract_transcript_text_from_obj(data)
    if speaker_prefix:
        extracted = f"{speaker_prefix}\n{extracted}".strip()
    if extracted:
        app.logger.info("Minutes transcript extracted from JSON length=%s token=%s", len(extracted), minute_token)
    else:
        app.logger.warning("Minutes transcript returned no usable text: token=%s payload=%s", minute_token, json.dumps(body)[:400])
    return (extracted[:MAX_MEETING_TRANSCRIPT_CHARS] if extracted else "", data)


def extract_open_ids_from_obj(obj) -> set[str]:
    ids: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                key_l = str(key).lower()
                if key_l in {"id", "open_id", "user_id", "speaker_id", "member_id", "owner_id"}:
                    if isinstance(value, str):
                        val = value.strip()
                        if val.startswith("ou_") and len(val) > 3:
                            ids.add(val)
                    elif isinstance(value, dict):
                        for sub_val in value.values():
                            if isinstance(sub_val, str):
                                sv = sub_val.strip()
                                if sv.startswith("ou_") and len(sv) > 3:
                                    ids.add(sv)
            for v in node.values():
                walk(v)
            return
        if isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)
    return ids


def normalize_person_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", " ", (name or "").lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def person_name_matches(candidate: str, target: str) -> bool:
    cand = normalize_person_name(candidate)
    targ = normalize_person_name(target)
    if not cand or not targ:
        return False
    if cand == targ:
        return True
    if cand in targ or targ in cand:
        return True

    cand_parts = set(cand.split())
    targ_parts = set(targ.split())
    if not cand_parts or not targ_parts:
        return False
    overlap = len(cand_parts & targ_parts)
    if overlap >= min(len(cand_parts), len(targ_parts)):
        return True
    if overlap >= 2:
        return True
    return False


def fetch_chat_member_directory(chat_id: str) -> list[dict]:
    members: list[dict] = []
    page_token = ""
    seen = set()

    for _ in range(5):
        params = {"member_id_type": "open_id", "page_size": 100}
        if page_token:
            params["page_token"] = page_token
        try:
            resp = requests.get(
                f"https://open.larkoffice.com/open-apis/im/v1/chats/{chat_id}/members",
                headers=lark_headers(),
                params=params,
                timeout=HTTP_TIMEOUT,
            )
            if resp.status_code >= 400:
                app.logger.warning("Chat member lookup failed chat_id=%s status=%s", chat_id, resp.status_code)
                break
            body = resp.json()
            if body.get("code", 0) != 0:
                app.logger.warning("Chat member lookup api error chat_id=%s code=%s msg=%s", chat_id, body.get("code"), body.get("msg"))
                break
            data = body.get("data") or {}
            items = data.get("items") or []
            for item in items:
                member_id = item.get("member_id") or {}
                if isinstance(member_id, str):
                    member_open_id = member_id.strip()
                else:
                    member_open_id = (
                        (member_id.get("open_id") if isinstance(member_id, dict) else "")
                        or (member_id.get("user_id") if isinstance(member_id, dict) else "")
                        or (member_id.get("id") if isinstance(member_id, dict) else "")
                        or (item.get("open_id") or "")
                    )
                member_open_id = (
                    (member_open_id or "").strip()
                )
                member_name = (item.get("name") or item.get("display_name") or item.get("member_name") or "").strip()
                key = (member_open_id, member_name)
                if key in seen:
                    continue
                seen.add(key)
                if member_open_id or member_name:
                    members.append({"open_id": member_open_id, "display_name": member_name})
            if not data.get("has_more"):
                break
            page_token = data.get("page_token") or ""
            if not page_token:
                break
        except Exception:
            app.logger.exception("Chat member lookup exception chat_id=%s", chat_id)
            break

    return members


def known_people_for_chat(chat_id: str, limit: int = 200) -> list[str]:
    names: list[str] = []
    seen = set()

    rows = db_query_all(
        """
        SELECT sender_name
        FROM messages
        WHERE chat_id = :chat_id
          AND sender_name IS NOT NULL
          AND sender_name <> ''
        ORDER BY created_at_ts DESC
        LIMIT :limit_rows
        """,
        {"chat_id": chat_id, "limit_rows": limit},
    )
    for row in rows:
        name = (row.get("sender_name") or "").strip()
        key = normalize_person_name(name)
        if name and key and key not in seen and key != "unknown":
            seen.add(key)
            names.append(name)

    for member in fetch_chat_member_directory(chat_id):
        name = (member.get("display_name") or "").strip()
        key = normalize_person_name(name)
        if name and key and key not in seen and key != "unknown":
            seen.add(key)
            names.append(name)

    return names[:200]


def lookup_open_id_by_email(email: str) -> str:
    target = (email or "").strip().lower()
    if not target or "@" not in target:
        return ""

    try:
        resp = requests.post(
            "https://open.larkoffice.com/open-apis/contact/v3/users/batch_get_id?user_id_type=open_id",
            headers=lark_headers(),
            json={"emails": [target]},
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code >= 400:
            app.logger.warning("Email lookup failed status=%s email=%s body=%s", resp.status_code, target, resp.text[:300])
            return ""
        body = resp.json()
        if body.get("code", 0) != 0:
            app.logger.warning(
                "Email lookup api error code=%s msg=%s email=%s",
                body.get("code"),
                body.get("msg"),
                target,
            )
            return ""
        data = body.get("data") or {}
        user_list = data.get("user_list") or []
        for item in user_list:
            resolved_id = (
                (item.get("open_id") or "").strip()
                or (item.get("user_id") or "").strip()
                or (item.get("id") or "").strip()
            )
            if ((item.get("email") or "").strip().lower() == target) and resolved_id:
                return resolved_id
        if user_list:
            resolved_id = (
                (user_list[0].get("open_id") or "").strip()
                or (user_list[0].get("user_id") or "").strip()
                or (user_list[0].get("id") or "").strip()
            )
            if resolved_id:
                return resolved_id
        app.logger.warning("Email lookup returned no matching user id for email=%s", target)
    except Exception:
        app.logger.exception("Email lookup exception email=%s", target)

    return ""


def batch_lookup_open_ids_by_email(emails: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    targets = list({(e or "").strip().lower() for e in emails if (e or "").strip() and "@" in (e or "")})
    if not targets:
        return result

    for i in range(0, len(targets), 50):
        batch = targets[i : i + 50]
        try:
            resp = requests.post(
                "https://open.larkoffice.com/open-apis/contact/v3/users/batch_get_id?user_id_type=open_id",
                headers=lark_headers(),
                json={"emails": batch},
                timeout=HTTP_TIMEOUT,
            )
            if resp.status_code >= 400:
                app.logger.warning("Batch email lookup failed status=%s", resp.status_code)
                continue
            body = resp.json()
            if body.get("code", 0) != 0:
                app.logger.warning("Batch email lookup api error code=%s msg=%s", body.get("code"), body.get("msg"))
                continue
            user_list = (body.get("data") or {}).get("user_list") or []
            for item in user_list:
                email = (item.get("email") or "").strip().lower()
                open_id = (
                    (item.get("open_id") or "").strip()
                    or (item.get("user_id") or "").strip()
                )
                if email and open_id:
                    result[email] = open_id
        except Exception:
            app.logger.exception("Batch email lookup exception")

    return result


def recent_sender_identities(chat_id: str | None = None, limit: int = 300) -> list[dict]:
    params = {"limit_rows": limit}
    sql = """
        SELECT sender_open_id, sender_name, MAX(created_at_ts) AS last_seen_ts
        FROM messages
        WHERE sender_open_id IS NOT NULL
          AND sender_open_id <> ''
          AND sender_name IS NOT NULL
          AND sender_name <> ''
          AND is_from_bot = 0
    """
    if chat_id:
        sql += "\n          AND chat_id = :chat_id"
        params["chat_id"] = chat_id
    sql += """
        GROUP BY sender_open_id, sender_name
        ORDER BY last_seen_ts DESC
        LIMIT :limit_rows
    """
    return db_query_all(sql, params)


def add_known_people_to_index(
    known_by_normalized_name: dict[str, dict],
    rows: list[dict],
    *,
    require_unique_names: bool = False,
) -> int:
    name_counts: dict[str, int] = {}
    if require_unique_names:
        for row in rows:
            key = normalize_person_name((row.get("sender_name") or row.get("display_name") or row.get("name") or "").strip())
            if key:
                name_counts[key] = name_counts.get(key, 0) + 1

    added = 0
    for row in rows:
        open_id = (row.get("sender_open_id") or row.get("open_id") or "").strip()
        name = (row.get("sender_name") or row.get("display_name") or row.get("name") or "").strip()
        if not open_id or not name:
            continue
        key = normalize_person_name(name)
        if not key:
            continue
        if require_unique_names and name_counts.get(key, 0) > 1 and key not in known_by_normalized_name:
            continue
        if key not in known_by_normalized_name:
            known_by_normalized_name[key] = {"open_id": open_id, "name": name}
            added += 1

    return added


def enrich_speaker_directory(speaker_directory: list[dict], chat_id: str) -> list[dict]:
    known_by_normalized_name: dict[str, dict] = {}

    # Source 1: message history for this chat (most reliable — actual senders)
    message_senders = recent_sender_identities(chat_id, limit=300)
    add_known_people_to_index(known_by_normalized_name, message_senders)

    # Source 2: identities learned from user mentions in this chat.
    mentioned_in_chat = recent_mentioned_identities(chat_id, limit=300)
    add_known_people_to_index(known_by_normalized_name, mentioned_in_chat)

    # Source 3: globally seen senders across all tracked chats (only when unambiguous).
    global_message_senders = recent_sender_identities(limit=600)
    add_known_people_to_index(known_by_normalized_name, global_message_senders, require_unique_names=True)

    # Source 4: identities learned from mentions across all chats (only when unambiguous).
    global_mentions = recent_mentioned_identities(limit=600)
    add_known_people_to_index(known_by_normalized_name, global_mentions, require_unique_names=True)

    # Source 5: cached user profiles (cross-chat — broadens reach)
    cached_profiles = db_query_all(
        """
        SELECT open_id, display_name
        FROM user_profile_cache
        WHERE display_name IS NOT NULL AND display_name <> ''
        ORDER BY cached_at_ts DESC
        LIMIT 300
        """,
    )
    add_known_people_to_index(known_by_normalized_name, cached_profiles)

    # Source 6: chat member API (may be restricted — bot often only sees itself)
    chat_members = fetch_chat_member_directory(chat_id)
    unnamed_resolved = 0
    for member in chat_members:
        member_open_id = (member.get("open_id") or "").strip()
        if not member_open_id:
            continue
        member_name = (member.get("display_name") or "").strip()
        if not member_name and unnamed_resolved < 40:
            unnamed_resolved += 1
            profile = get_user_profile(member_open_id)
            member_name = (profile.get("display_name") or "").strip()
        if not member_name:
            continue
        add_known_people_to_index(
            known_by_normalized_name,
            [{"open_id": member_open_id, "display_name": member_name}],
        )

    app.logger.warning(
        "Enrichment: chat_id=%s msg_senders=%s chat_mentions=%s global_msg_senders=%s global_mentions=%s cached_profiles=%s chat_members=%s total_known=%s",
        chat_id,
        len(message_senders),
        len(mentioned_in_chat),
        len(global_message_senders),
        len(global_mentions),
        len(cached_profiles),
        len(chat_members),
        len(known_by_normalized_name),
    )

    # Step 1: fill open_ids on existing speaker_directory entries via name match
    for entry in speaker_directory:
        if (entry.get("open_id") or "").strip():
            continue
        entry_name = (entry.get("name") or "").strip()
        if not entry_name:
            continue
        for person in known_by_normalized_name.values():
            if person_name_matches(entry_name, person["name"]):
                entry["open_id"] = person["open_id"]
                break

    # Step 2: batch resolve emails still missing open_ids
    emails_to_resolve = []
    for entry in speaker_directory:
        if (entry.get("open_id") or "").strip():
            continue
        email = (entry.get("email") or "").strip().lower()
        if email and "@" in email:
            emails_to_resolve.append(email)

    if emails_to_resolve:
        email_to_open_id = batch_lookup_open_ids_by_email(emails_to_resolve)
        for entry in speaker_directory:
            if (entry.get("open_id") or "").strip():
                continue
            email = (entry.get("email") or "").strip().lower()
            if email in email_to_open_id:
                entry["open_id"] = email_to_open_id[email]

    # Step 3: add all known people to directory so Gemini owner names can match
    existing_names = {normalize_person_name(e.get("name") or "") for e in speaker_directory}
    existing_open_ids = {(e.get("open_id") or "").strip() for e in speaker_directory if (e.get("open_id") or "").strip()}
    for person in known_by_normalized_name.values():
        key = normalize_person_name(person["name"])
        if key in existing_names:
            continue
        if person["open_id"] in existing_open_ids:
            continue
        speaker_directory.append({
            "name": person["name"],
            "email": "",
            "open_id": person["open_id"],
        })

    return speaker_directory


def lookup_owner_open_id(chat_id: str, owner_name: str) -> str:
    target = (owner_name or "").strip()
    if not target:
        return ""

    for row in recent_sender_identities(chat_id, limit=200):
        sender_name = (row.get("sender_name") or "").strip()
        if person_name_matches(sender_name, target):
            return (row.get("sender_open_id") or "").strip()

    for row in recent_mentioned_identities(chat_id, limit=200):
        display_name = (row.get("display_name") or "").strip()
        if person_name_matches(display_name, target):
            return (row.get("open_id") or "").strip()

    global_known_by_name: dict[str, dict] = {}
    add_known_people_to_index(global_known_by_name, recent_sender_identities(limit=500), require_unique_names=True)
    for person in global_known_by_name.values():
        if person_name_matches(person["name"], target):
            return person["open_id"]

    global_mentions_by_name: dict[str, dict] = {}
    add_known_people_to_index(global_mentions_by_name, recent_mentioned_identities(limit=500), require_unique_names=True)
    for person in global_mentions_by_name.values():
        if person_name_matches(person["name"], target):
            return person["open_id"]

    cached = db_query_all(
        """
        SELECT open_id, display_name
        FROM user_profile_cache
        WHERE display_name IS NOT NULL
          AND display_name <> ''
        ORDER BY cached_at_ts DESC
        LIMIT 200
        """
    )
    for row in cached:
        display_name = (row.get("display_name") or "").strip()
        if person_name_matches(display_name, target):
            return (row.get("open_id") or "").strip()

    for member in fetch_chat_member_directory(chat_id):
        display_name = (member.get("display_name") or "").strip()
        if person_name_matches(display_name, target):
            return (member.get("open_id") or "").strip()

    return ""


def extract_transcript_speaker_emails(transcript_text: str) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    if not transcript_text:
        return out

    patterns = [
        re.compile(
            r"([^\n:<>]{1,120}?)\s*<([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})>",
            re.IGNORECASE,
        ),
        re.compile(
            r"([^\n:()]{1,120}?)\s*\(([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})\)",
            re.IGNORECASE,
        ),
        re.compile(
            r"([^\n:]{1,120}?)\s+([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})\s*:",
            re.IGNORECASE,
        ),
    ]

    lines = [(line or "").strip() for line in (transcript_text or "").splitlines()]
    for idx, line_text in enumerate(lines):
        if not line_text:
            continue
        for pattern in patterns:
            for match in pattern.finditer(line_text):
                speaker_name = sanitize_text((match.group(1) or "").strip(" -|"))
                speaker_email = (match.group(2) or "").strip().lower()
                if not speaker_name or not speaker_email:
                    continue
                key = (normalize_person_name(speaker_name), speaker_email)
                if not key[0] or key in seen:
                    continue
                seen.add(key)
                out.append({"name": speaker_name, "email": speaker_email})

        speaker_label_patterns = [
            re.compile(
                r"^\s*(?:\[[^\]]{1,24}\]\s*)?([A-Za-z][A-Za-z0-9 .'\-]{0,79}|[\u4e00-\u9fff]{1,24})\s*:\s+\S",
                re.IGNORECASE,
            ),
            re.compile(
                r"^\s*(?:\d{1,2}:\d{2}(?::\d{2})?\s+)?([A-Za-z][A-Za-z0-9 .'\-]{0,79}|[\u4e00-\u9fff]{1,24})\s*:\s+\S",
                re.IGNORECASE,
            ),
        ]
        for pattern in speaker_label_patterns:
            label_match = pattern.match(line_text)
            if not label_match:
                continue
            speaker_name = sanitize_text((label_match.group(1) or "").strip(" -|"))
            if not speaker_name:
                continue
            key = (normalize_person_name(speaker_name), "")
            if key[0] and key not in seen:
                seen.add(key)
                out.append({"name": speaker_name, "email": ""})
                break

        standalone_email = re.fullmatch(
            r"([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})",
            line_text,
            re.IGNORECASE,
        )
        if standalone_email:
            speaker_email = standalone_email.group(1).strip().lower()
            speaker_name = ""
            for prev_idx in range(idx - 1, max(-1, idx - 4), -1):
                candidate = lines[prev_idx].strip()
                if not candidate:
                    continue
                if "@" in candidate:
                    continue
                if len(candidate) > 120:
                    continue
                speaker_name = sanitize_text(candidate.strip(" -|"))
                if speaker_name:
                    break
            if speaker_name:
                key = (normalize_person_name(speaker_name), speaker_email)
                if key[0] and key not in seen:
                    seen.add(key)
                    out.append({"name": speaker_name, "email": speaker_email})

    return out


def infer_owner_email_from_directory(owner_name: str, speaker_directory: list[dict] | None) -> str:
    target = (owner_name or "").strip()
    if not target or not speaker_directory:
        return ""

    email_match = re.search(r"([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})", target, re.IGNORECASE)
    if email_match:
        return email_match.group(1).strip().lower()

    normalized_target = normalize_person_name(target)
    if not normalized_target:
        return ""

    exact_match_email = ""
    fuzzy_match_email = ""
    for item in speaker_directory:
        speaker_name = (item.get("name") or "").strip()
        speaker_email = (item.get("email") or "").strip().lower()
        if not speaker_name or not speaker_email:
            continue
        normalized_speaker = normalize_person_name(speaker_name)
        if normalized_speaker == normalized_target:
            exact_match_email = speaker_email
            break
        if not fuzzy_match_email and person_name_matches(speaker_name, target):
            fuzzy_match_email = speaker_email

    return exact_match_email or fuzzy_match_email


def infer_owner_open_id_from_directory(owner_name: str, owner_email: str, speaker_directory: list[dict] | None) -> str:
    target_name = (owner_name or "").strip()
    target_email = (owner_email or "").strip().lower()
    if not speaker_directory:
        return ""

    if not target_email:
        email_match = re.search(r"([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})", target_name, re.IGNORECASE)
        if email_match:
            target_email = email_match.group(1).strip().lower()

    if target_email:
        for item in speaker_directory:
            item_email = (item.get("email") or "").strip().lower()
            item_open_id = (
                (item.get("open_id") or "").strip()
                or (item.get("user_id") or "").strip()
                or (item.get("id") or "").strip()
            )
            if item_email and item_email == target_email and item_open_id:
                return item_open_id

    normalized_target = normalize_person_name(target_name)
    if not normalized_target:
        return ""

    exact_match_open_id = ""
    fuzzy_match_open_id = ""
    for item in speaker_directory:
        item_name = (item.get("name") or "").strip()
        item_open_id = (
            (item.get("open_id") or "").strip()
            or (item.get("user_id") or "").strip()
            or (item.get("id") or "").strip()
        )
        if not item_name or not item_open_id:
            continue
        normalized_item = normalize_person_name(item_name)
        if normalized_item == normalized_target:
            exact_match_open_id = item_open_id
            break
        if not fuzzy_match_open_id and person_name_matches(item_name, target_name):
            fuzzy_match_open_id = item_open_id

    return exact_match_open_id or fuzzy_match_open_id


def build_owner_candidates(speaker_directory: list[dict] | None, known_people: list[str] | None = None) -> list[dict]:
    candidates: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for item in speaker_directory or []:
        name = (item.get("name") or "").strip()
        email = (item.get("email") or "").strip().lower()
        key = (normalize_person_name(name), email)
        if not key[0] or key in seen:
            continue
        seen.add(key)
        candidates.append({"name": name, "email": email})

    for name in known_people or []:
        clean_name = (name or "").strip()
        key = (normalize_person_name(clean_name), "")
        if not key[0] or key in seen:
            continue
        seen.add(key)
        candidates.append({"name": clean_name, "email": ""})

    return candidates[:300]


def canonicalize_owner_identity(
    owner_name: str,
    owner_email: str,
    speaker_directory: list[dict] | None,
    known_people: list[str] | None = None,
) -> tuple[str, str]:
    raw_name = (owner_name or "").strip()
    raw_email = (owner_email or "").strip().lower()
    if not raw_email and raw_name:
        email_match = re.search(r"([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})", raw_name, re.IGNORECASE)
        if email_match:
            raw_email = email_match.group(1).strip().lower()
            stripped_name = sanitize_text(raw_name.replace(email_match.group(1), "").strip("()<> -"))
            if stripped_name:
                raw_name = stripped_name

    candidates = build_owner_candidates(speaker_directory, known_people)
    if not candidates:
        return raw_name, raw_email

    if raw_email:
        email_matches = [c for c in candidates if (c.get("email") or "").strip().lower() == raw_email]
        if len(email_matches) == 1:
            matched_name = (email_matches[0].get("name") or "").strip()
            return matched_name or raw_name, raw_email

    normalized_target = normalize_person_name(raw_name)
    if not normalized_target:
        return raw_name, raw_email

    exact_matches = [c for c in candidates if normalize_person_name(c.get("name") or "") == normalized_target]
    if len(exact_matches) == 1:
        matched_name = (exact_matches[0].get("name") or "").strip()
        matched_email = (exact_matches[0].get("email") or "").strip().lower() or raw_email
        return matched_name or raw_name, matched_email

    fuzzy_matches = [c for c in candidates if person_name_matches(c.get("name") or "", raw_name)]
    unique_names = {(c.get("name") or "").strip() for c in fuzzy_matches if (c.get("name") or "").strip()}
    if len(unique_names) == 1 and fuzzy_matches:
        matched = fuzzy_matches[0]
        matched_name = (matched.get("name") or "").strip()
        matched_email = (matched.get("email") or "").strip().lower() or raw_email
        return matched_name or raw_name, matched_email

    return raw_name, raw_email


def normalize_meeting_next_step_owners(
    parsed: dict,
    speaker_directory: list[dict] | None,
    known_people: list[str] | None = None,
):
    next_steps = parsed.get("next_steps")
    if not isinstance(next_steps, list):
        return

    for item in next_steps:
        if not isinstance(item, dict):
            continue
        original_name = (item.get("owner_name") or "").strip()
        original_email = (item.get("owner_email") or "").strip().lower()
        canonical_name, canonical_email = canonicalize_owner_identity(
            original_name,
            original_email,
            speaker_directory,
            known_people=known_people,
        )
        if canonical_name and canonical_name != original_name:
            item["owner_name"] = canonical_name
            app.logger.info("Canonicalized meeting owner name: raw=%s canonical=%s", original_name, canonical_name)
        if canonical_email and canonical_email != original_email:
            item["owner_email"] = canonical_email


def render_owner_reference(chat_id: str, owner_name: str, owner_email: str = "", speaker_directory: list[dict] | None = None) -> str:
    raw_name = (owner_name or "").strip()
    clean_name = raw_name
    clean_email = (owner_email or "").strip()
    if not clean_email:
        email_match = re.search(r"([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})", raw_name, re.IGNORECASE)
        if email_match:
            clean_email = email_match.group(1)
            clean_name = sanitize_text(raw_name.replace(clean_email, "").strip("()<> -")) or clean_name
    direct_open_id = infer_owner_open_id_from_directory(clean_name or raw_name, clean_email, speaker_directory)
    if direct_open_id:
        label = clean_name or raw_name or clean_email or "there"
        return f'<at user_id="{direct_open_id}">{label}</at>'
    if not clean_email:
        clean_email = infer_owner_email_from_directory(clean_name or raw_name, speaker_directory)
    if clean_email:
        email_open_id = lookup_open_id_by_email(clean_email)
        if email_open_id:
            label = clean_name or clean_email
            return f'<at user_id="{email_open_id}">{label}</at>'
        app.logger.warning("Owner email could not be resolved for tag: name=%s email=%s", clean_name or raw_name, clean_email)
    if not clean_name:
        return "Unassigned"
    open_id = lookup_owner_open_id(chat_id, clean_name)
    if open_id:
        return f'<at user_id="{open_id}">{clean_name}</at>'
    app.logger.warning("Owner name could not be resolved for tag: name=%s", clean_name)
    return clean_name


def analyze_meeting_with_gemini(
    request_text: str,
    minute_url: str,
    minute_meta: dict,
    transcript_text: str,
    media_bytes: bytes | None,
    media_mime_type: str | None,
    owner_candidates: list[dict],
    known_people: list[str],
    learned_terms: list[str],
) -> dict:
    user_payload = {
        "request": request_text,
        "minute_url": minute_url,
        "minute_meta": minute_meta,
        "transcript_excerpt": transcript_text,
        "owner_candidates": owner_candidates,
        "known_people": known_people,
        "learned_terms": learned_terms,
    }
    parts = [
        types.Part.from_text(text=json.dumps(user_payload, ensure_ascii=False)),
    ]
    if media_bytes and media_mime_type:
        parts.append(types.Part.from_bytes(data=media_bytes, mime_type=media_mime_type))

    config = types.GenerateContentConfig(
        system_instruction=MEETING_SUMMARY_SYSTEM_PROMPT,
        response_mime_type="application/json",
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[types.Content(role="user", parts=parts)],
        config=config,
    )
    raw = (response.text or "").strip()
    return json.loads(raw)


def format_meeting_reply(
    chat_id: str,
    mode: str,
    parsed: dict,
    meeting_title: str = "",
    speaker_directory: list[dict] | None = None,
) -> str:
    summary_bullets = []
    for idx, item in enumerate(parsed.get("summary_bullets") or [], start=1):
        text_item = (item or "").strip()
        if text_item:
            summary_bullets.append(f"{idx}. {text_item}")

    next_step_lines = []
    for idx, item in enumerate(parsed.get("next_steps") or [], start=1):
        task = (item.get("task") or "").strip()
        if not task:
            continue
        owner_ref = render_owner_reference(
            chat_id,
            item.get("owner_name") or "",
            item.get("owner_email") or "",
            speaker_directory=speaker_directory,
        )
        if owner_ref == "Unassigned":
            next_step_lines.append(f"{idx}. {task}")
        else:
            next_step_lines.append(f"{idx}. {owner_ref}: {task}")

    header = "Next steps:" if mode == "next_steps" else "Meeting summary:"

    lines = [header]
    if mode != "next_steps" and summary_bullets:
        lines.extend(summary_bullets)
    if next_step_lines:
        if mode != "next_steps":
            if summary_bullets:
                lines.append("")
            lines.append("Next steps:")
        lines.extend(next_step_lines)
    return "\n".join(lines)


def parse_meeting_datetime(value, tz_name: str = "UTC") -> datetime | None:
    if value in (None, ""):
        return None
    tz = timezone.utc
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc

    if isinstance(value, (int, float)):
        num = float(value)
        if num > 1_000_000_000_000:
            num = num / 1000.0
        return datetime.fromtimestamp(num, tz=tz)

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if re.fullmatch(r"\d{10,16}", raw):
            num = float(raw)
            if num > 1_000_000_000_000:
                num = num / 1000.0
            return datetime.fromtimestamp(num, tz=tz)
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=tz)
            return parsed.astimezone(tz)
        except Exception:
            return None

    return None


def build_meeting_post_title(minute_meta: dict) -> str:
    containers = [
        minute_meta or {},
        (minute_meta.get("minute") or {}) if isinstance(minute_meta, dict) else {},
        (minute_meta.get("item") or {}) if isinstance(minute_meta, dict) else {},
    ]

    base_title = ""
    tz_name = "UTC"
    for container in containers:
        if not isinstance(container, dict):
            continue
        if not base_title:
            base_title = (container.get("title") or "").strip()
        if tz_name == "UTC":
            candidate_tz = (container.get("timezone") or container.get("time_zone") or container.get("tz") or "").strip()
            if candidate_tz:
                tz_name = candidate_tz

    date_value = None
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in ["start_time", "meeting_start_time", "begin_time", "create_time", "created_at", "start_time_ms"]:
            if container.get(key) not in (None, ""):
                date_value = container.get(key)
                break
        if date_value not in (None, ""):
            break

    date_label = ""
    parsed_dt = parse_meeting_datetime(date_value, tz_name=tz_name)
    if parsed_dt:
        date_label = f"{parsed_dt.day} {parsed_dt.strftime('%b')}"

    clean_title = sanitize_text(base_title)
    if clean_title and date_label and f"({date_label})" not in clean_title:
        return f"{clean_title} ({date_label})"
    if clean_title:
        return clean_title
    if date_label:
        return f"Meeting ({date_label})"
    return "Meeting"


def extract_meeting_owner_candidates(minute_meta: dict) -> list[dict]:
    if not isinstance(minute_meta, dict):
        return []

    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    resolved_ids: dict[tuple[str, str], dict] = {}
    owner_hints = ("owner", "host", "organizer", "organiser", "creator")
    name_keys = {
        "owner_name", "host_name", "organizer_name", "organiser_name", "creator_name",
        "name", "user_name", "display_name", "full_name", "en_name", "nick_name", "nickname",
    }

    def collect_id_candidates(node, max_depth: int = 2) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        local_seen: set[tuple[str, str]] = set()

        def maybe_add(identifier, id_type: str):
            if not isinstance(identifier, str):
                return
            clean_id = identifier.strip()
            if not clean_id:
                return
            key = (clean_id, id_type)
            if key in local_seen:
                return
            local_seen.add(key)
            candidates.append(key)

        def walk(value, depth: int):
            if depth < 0:
                return
            if isinstance(value, dict):
                member_id = value.get("member_id")
                if isinstance(member_id, dict):
                    maybe_add(member_id.get("open_id"), "open_id")
                    maybe_add(member_id.get("user_id"), "user_id")
                    maybe_add(member_id.get("union_id"), "union_id")
                    maybe_add(member_id.get("id"), "open_id")
                elif isinstance(member_id, str):
                    maybe_add(member_id, "open_id")

                for key, item in value.items():
                    key_l = str(key).lower()
                    if key_l == "open_id":
                        maybe_add(item, "open_id")
                    elif key_l in {"user_id", "owner_user_id", "host_user_id", "creator_user_id"}:
                        maybe_add(item, "user_id")
                    elif key_l == "union_id":
                        maybe_add(item, "union_id")
                    elif key_l in {"id", "owner_id", "host_id", "creator_id"}:
                        maybe_add(item, "open_id")

                if depth > 0:
                    for item in value.values():
                        walk(item, depth - 1)
            elif isinstance(value, list) and depth > 0:
                for item in value:
                    walk(item, depth - 1)

        walk(node, max_depth)
        return candidates

    def maybe_add(node, parent_key: str = "", context_hint: bool = False):
        if not isinstance(node, dict):
            return

        parent_hint = (parent_key or "").lower()
        node_keys = {str(k).lower() for k in node.keys()}
        has_context = context_hint or any(hint in parent_hint for hint in owner_hints) or any(
            hint in key for key in node_keys for hint in owner_hints
        )
        if not has_context:
            return

        owner_name = sanitize_text(first_nested_string(node, name_keys, max_depth=2))
        owner_open_id = ""
        for identifier, id_type in collect_id_candidates(node, max_depth=2):
            cache_key = (identifier, id_type)
            if cache_key not in resolved_ids:
                resolved_ids[cache_key] = fetch_lark_user_identity(identifier, id_type)
            identity = resolved_ids[cache_key]
            if not owner_open_id and (identity.get("open_id") or "").strip():
                owner_open_id = (identity.get("open_id") or "").strip()
            if not owner_name and (identity.get("display_name") or "").strip():
                owner_name = (identity.get("display_name") or "").strip()
            if owner_open_id and owner_name:
                break

        normalized_name = normalize_person_name(owner_name)
        item_key = (owner_open_id, normalized_name)
        if item_key in seen:
            return
        if not owner_open_id and not normalized_name:
            return
        seen.add(item_key)
        out.append({"name": owner_name, "open_id": owner_open_id})

    def walk(node, parent_key: str = "", context_hint: bool = False):
        if isinstance(node, dict):
            node_keys = {str(k).lower() for k in node.keys()}
            node_hint = context_hint or any(hint in key for key in node_keys for hint in owner_hints)
            maybe_add(node, parent_key=parent_key, context_hint=node_hint)
            for key, value in node.items():
                child_key = str(key).lower()
                child_hint = node_hint or any(hint in child_key for hint in owner_hints)
                walk(value, parent_key=child_key, context_hint=child_hint)
        elif isinstance(node, list):
            for item in node:
                walk(item, parent_key=parent_key, context_hint=context_hint)

    walk(minute_meta)
    return out


def minute_owned_by_requester(minute_meta: dict, requester_open_id: str, requester_name: str) -> bool:
    clean_requester_id = (requester_open_id or "").strip()
    clean_requester_name = sanitize_text(requester_name or "")
    for candidate in extract_meeting_owner_candidates(minute_meta):
        owner_open_id = (candidate.get("open_id") or "").strip()
        owner_name = (candidate.get("name") or "").strip()
        if clean_requester_id and owner_open_id and owner_open_id == clean_requester_id:
            return True
        if clean_requester_name and owner_name and person_name_matches(owner_name, clean_requester_name):
            return True
    return False


def find_last_requester_owned_meeting_url(
    chat_id: str,
    requester_open_id: str,
    requester_name: str,
    scan_limit: int = 160,
    candidate_limit: int = 12,
) -> str:
    if not requester_open_id and not (requester_name or "").strip():
        return ""

    for meeting_url in recent_meeting_urls(chat_id, scan_limit=scan_limit, unique_limit=candidate_limit):
        minute_token = extract_minutes_token(meeting_url)
        if not minute_token:
            continue
        try:
            minute_meta = fetch_lark_minute_meta(minute_token)
        except Exception:
            app.logger.warning("Requester-owned meeting lookup skipped meta fetch: token=%s", minute_token)
            continue
        if minute_owned_by_requester(minute_meta, requester_open_id, requester_name):
            app.logger.info(
                "Requester-owned meeting selected: chat_id=%s token=%s requester=%s",
                chat_id,
                minute_token,
                requester_name,
            )
            return meeting_url
    return ""


def choose_meeting_url_for_request(chat_id: str, request_text: str, requester_open_id: str, requester_name: str) -> str:
    direct_url = extract_minutes_url(request_text)
    if direct_url:
        return direct_url

    if prefers_requester_owned_recent_meeting(request_text):
        owned_url = find_last_requester_owned_meeting_url(chat_id, requester_open_id, requester_name)
        if owned_url:
            return owned_url

    return find_last_meeting_url(chat_id)


def build_meeting_reply(chat_id: str, request_text: str, minute_url: str, mode: str) -> tuple[str, str]:
    minute_token = extract_minutes_token(minute_url)
    if not minute_token:
        raise RuntimeError("invalid meeting transcript link")

    minute_meta = {}
    try:
        minute_meta = fetch_lark_minute_meta(minute_token)
    except Exception:
        app.logger.exception("Minutes meta unavailable; continuing with transcript/media only")

    transcript_text, transcript_data = fetch_lark_minute_transcript(minute_token)
    if not transcript_text and minute_meta:
        transcript_text = extract_transcript_text_from_obj(minute_meta)
    public_page_text = ""
    if not transcript_text:
        public_page_text = fetch_public_minute_page_text(minute_url)
        if public_page_text:
            transcript_text = extract_transcript_text_from_public_html(public_page_text)
            if transcript_text:
                app.logger.info(
                    "Minutes transcript extracted from public page length=%s token=%s",
                    len(transcript_text),
                    minute_token,
                )
    media_bytes = None
    media_mime_type = None

    if len(transcript_text) < 800:
        media_bytes, media_mime_type = fetch_lark_minute_media(minute_token)

    if not transcript_text and not media_bytes:
        raise MeetingTranscriptAccessError("meeting transcript is not accessible yet")

    # --- Build speaker directory from meeting data sources ---
    # Source 1: speaker emails from transcript text
    transcript_directory = extract_transcript_speaker_emails(transcript_text)
    # Source 2: participant info from statistics API
    statistics_data = fetch_lark_minute_statistics(minute_token)
    participant_directory = extract_participant_directory_from_obj(statistics_data, allow_name_only=True)
    if not participant_directory:
        participant_directory = fetch_public_minute_page_people(minute_url, page_text=public_page_text or None)
    # Source 3: participant info from transcript JSON and minute metadata
    meta_participants = extract_participant_directory_from_obj(minute_meta, allow_name_only=True) if minute_meta else []
    transcript_participants = extract_participant_directory_from_obj(transcript_data, allow_name_only=True) if transcript_data else []
    if not participant_directory and not meta_participants and not transcript_participants:
        fallback_names = []
        for source_names in [
            extract_name_like_values_from_obj(statistics_data, limit=20),
            extract_name_like_values_from_obj(minute_meta, limit=20) if minute_meta else [],
            extract_name_like_values_from_obj(transcript_data, limit=20) if transcript_data else [],
            extract_name_like_values_from_public_html(public_page_text, limit=20) if public_page_text else [],
        ]:
            for name in source_names:
                key = normalize_person_name(name)
                if not key:
                    continue
                if any(normalize_person_name(existing.get("name") or "") == key for existing in fallback_names):
                    continue
                fallback_names.append({"name": name, "email": "", "open_id": ""})
        if fallback_names:
            participant_directory = fallback_names
            app.logger.warning(
                "Meeting participant name fallback used: token=%s count=%s samples=%s",
                minute_token,
                len(fallback_names),
                [item["name"] for item in fallback_names[:8]],
            )
        else:
            app.logger.warning(
                "Meeting participant extraction empty: token=%s stats_name_samples=%s meta_name_samples=%s transcript_name_samples=%s public_name_samples=%s",
                minute_token,
                extract_name_like_values_from_obj(statistics_data, limit=8),
                extract_name_like_values_from_obj(minute_meta, limit=8) if minute_meta else [],
                extract_name_like_values_from_obj(transcript_data, limit=8) if transcript_data else [],
                extract_name_like_values_from_public_html(public_page_text, limit=8) if public_page_text else [],
            )

    speaker_directory = merge_people_directories(
        participant_directory, transcript_directory, meta_participants, transcript_participants,
    )

    # Source 4: extract raw open_ids from meeting JSON structures and resolve to profiles
    meeting_open_ids: set[str] = set()
    if transcript_data:
        meeting_open_ids |= extract_open_ids_from_obj(transcript_data)
    if minute_meta:
        meeting_open_ids |= extract_open_ids_from_obj(minute_meta)
    if LARK_BOT_OPEN_ID:
        meeting_open_ids.discard(LARK_BOT_OPEN_ID)

    existing_ids = {(e.get("open_id") or "").strip() for e in speaker_directory if (e.get("open_id") or "").strip()}
    resolved_from_meeting = 0
    for open_id in meeting_open_ids:
        if open_id in existing_ids:
            continue
        profile = get_user_profile(open_id)
        name = (profile.get("display_name") or "").strip()
        if name:
            speaker_directory.append({"name": name, "email": "", "open_id": open_id})
            existing_ids.add(open_id)
            resolved_from_meeting += 1

    app.logger.warning(
        "Meeting speaker directory: token=%s text_people=%s stats_participants=%s meta=%s transcript_json=%s meeting_ids=%s resolved=%s total=%s",
        minute_token,
        len(transcript_directory),
        len(participant_directory),
        len(meta_participants),
        len(transcript_participants),
        len(meeting_open_ids),
        resolved_from_meeting,
        len(speaker_directory),
    )

    # Fallback: enrich with chat message history and cached profiles
    speaker_directory = enrich_speaker_directory(speaker_directory, chat_id)
    enriched_with_id = sum(1 for e in speaker_directory if (e.get("open_id") or "").strip())
    app.logger.warning(
        "Speaker directory final: token=%s total=%s with_open_id=%s",
        minute_token,
        len(speaker_directory),
        enriched_with_id,
    )

    known_people = known_people_for_chat(chat_id)
    for item in speaker_directory:
        speaker_name = (item.get("name") or "").strip()
        if not speaker_name:
            continue
        key = normalize_person_name(speaker_name)
        if key and all(normalize_person_name(existing) != key for existing in known_people):
            known_people.append(speaker_name)
    owner_candidates = build_owner_candidates(speaker_directory, known_people)
    learned_terms = load_learned_terms(chat_id)
    parsed = analyze_meeting_with_gemini(
        request_text=request_text,
        minute_url=minute_url,
        minute_meta=minute_meta,
        transcript_text=transcript_text,
        media_bytes=media_bytes,
        media_mime_type=media_mime_type,
        owner_candidates=owner_candidates,
        known_people=known_people,
        learned_terms=learned_terms,
    )
    normalize_meeting_next_step_owners(parsed, speaker_directory, known_people)

    title = build_meeting_post_title(minute_meta)
    reply_text = format_meeting_reply(
        chat_id,
        mode,
        parsed,
        meeting_title=title,
        speaker_directory=speaker_directory,
    )
    return reply_text, title


def send_missing_history_message(reply_to_message_id: str, chat_id: str):
    if THOMAS_OPEN_ID:
        text = "I don't have access to historical messages yet, let's check what's going on!"
        mention_id = THOMAS_OPEN_ID
    else:
        text = "I don't have access to historical messages yet, Thomas let's check what's going on!"
        mention_id = None
    msg_id = send_lark_text_reply(
        reply_to_message_id,
        text,
        mention_open_id=mention_id,
        mention_name=THOMAS_DISPLAY_NAME,
    )
    remember_bot_message(msg_id, chat_id)
    save_bot_text(chat_id, text, None, reply_to_message_id, event_message_id=msg_id)


def looks_like_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def find_first_media_url(obj):
    if isinstance(obj, dict):
        for _, value in obj.items():
            found = find_first_media_url(value)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_first_media_url(item)
            if found:
                return found
    elif isinstance(obj, str):
        s = obj.strip()
        if s.startswith("http://") or s.startswith("https://"):
            if re.search(r"\.(gif|png|jpg|jpeg|webp)(\?|$)", s, re.I):
                return s
    return None


def extract_media_urls_from_tenor_like(data: dict, max_results: int = 8) -> list[str]:
    results = data.get("results")
    if not isinstance(results, list) or not results:
        return []
    out = []
    # Prefer smaller variants first to satisfy Lark image upload limits.
    preferred_order = ["tinygif", "nanogif", "gif", "mediumgif", "tinywebp", "webp"]
    for item in results[:max_results]:
        media_formats = (item or {}).get("media_formats") or {}
        for key in preferred_order:
            candidate = media_formats.get(key) or {}
            url = candidate.get("url")
            if isinstance(url, str) and url.startswith(("http://", "https://")) and url not in out:
                out.append(url)
        fallback = find_first_media_url(item)
        if fallback and fallback not in out:
            out.append(fallback)
    return out


def search_klipy_stickers(query: str) -> list[str]:
    if not KLIPY_API_KEY:
        return []

    candidates: list[str] = []
    seen = set()

    def add_candidates(urls: list[str]):
        for u in urls:
            if u and u not in seen:
                seen.add(u)
                candidates.append(u)

    # Strategy 1: Tenor-compatible search endpoint used by KLIPY migration docs.
    try:
        resp = requests.get(
            "https://api.klipy.com/v2/search",
            params={"q": query, "key": KLIPY_API_KEY, "limit": 10, "media_filter": "gif"},
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code < 400:
            data = resp.json()
            urls = extract_media_urls_from_tenor_like(data, max_results=10)
            if urls:
                add_candidates(urls)
                app.logger.info("Klipy stickers found via /v2/search count=%s", len(urls))
        else:
            app.logger.warning("Klipy /v2/search failed status=%s body=%s", resp.status_code, resp.text[:200])
    except Exception:
        app.logger.exception("Klipy /v2/search request error")

    # Strategy 2: caller-provided URL (POST JSON).
    headers = {
        "Authorization": f"Bearer {KLIPY_API_KEY}",
        "X-API-Key": KLIPY_API_KEY,
        "X-KLIPY-API-KEY": KLIPY_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"query": query, "q": query, "limit": 10}
    try:
        resp = requests.post(KLIPY_SEARCH_URL, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
        if resp.status_code < 400:
            data = resp.json()
            urls = extract_media_urls_from_tenor_like(data, max_results=10)
            if not urls:
                single = find_first_media_url(data)
                urls = [single] if single else []
            if urls:
                add_candidates(urls)
                app.logger.info("Klipy stickers found via configured URL count=%s", len(urls))
        else:
            app.logger.warning("Klipy configured URL failed status=%s body=%s", resp.status_code, resp.text[:200])
    except Exception:
        app.logger.exception("Klipy configured URL request error")

    return candidates


def load_recent_context_for_sticker(chat_id: str, exclude_event_message_id: str | None, limit: int = 20) -> list[str]:
    rows = db_query_all(
        """
        SELECT sender_name, is_from_bot, message_type, text_content, created_at_ts, event_message_id
        FROM messages
        WHERE chat_id = :chat_id
          AND (event_message_id IS NULL OR event_message_id <> :exclude_event_message_id)
          AND (text_content IS NOT NULL AND text_content <> '')
        ORDER BY created_at_ts DESC
        LIMIT :limit_rows
        """,
        {
            "chat_id": chat_id,
            "exclude_event_message_id": exclude_event_message_id or "",
            "limit_rows": limit,
        },
    )
    lines = []
    for row in reversed(rows):
        name = (row.get("sender_name") or "").strip() or ("Sticksy" if int(row.get("is_from_bot") or 0) == 1 else "Participant")
        text = (row.get("text_content") or "").strip()
        if text:
            lines.append(f"{name}: {text}")
    return lines


def build_sticker_search_keyword(user_query: str, context_lines: list[str]) -> str:
    raw = sanitize_text(user_query)
    if not raw:
        return ""
    try:
        config = types.GenerateContentConfig(
            system_instruction=STICKER_QUERY_SYSTEM_PROMPT,
            response_mime_type="application/json",
        )
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=json.dumps(
                {
                    "user_query": raw,
                    "recent_context": context_lines[-20:],
                },
                ensure_ascii=False,
            ),
            config=config,
        )
        parsed = json.loads((resp.text or "{}").strip())
        keyword = sanitize_text(parsed.get("query") or parsed.get("keyword") or "")
        if keyword:
            return keyword[:80]
    except Exception:
        app.logger.exception("Sticker keyword rewrite failed; fallback to raw")
    return raw[:80]


def send_sticker_reply(reply_to_message_id: str, chat_id: str, query: str, current_message_id: str | None = None):
    context_lines = load_recent_context_for_sticker(chat_id, current_message_id, limit=20)
    sticker_query = build_sticker_search_keyword(query, context_lines)
    if not sticker_query:
        app.logger.info("No sticker keyword for query=%s", query)
        return
    app.logger.info("Sticker keyword rewritten: raw=%s rewritten=%s", query, sticker_query)

    sticker_urls = search_klipy_stickers(sticker_query)
    if not sticker_urls:
        app.logger.info("No sticker result for query=%s rewritten=%s", query, sticker_query)
        return

    for idx, sticker_url in enumerate(sticker_urls[:8], start=1):
        try:
            media = requests.get(sticker_url, timeout=HTTP_TIMEOUT)
            if media.status_code >= 400:
                app.logger.warning("Sticker media download failed status=%s url=%s", media.status_code, sticker_url)
                continue
            content_length = int(media.headers.get("Content-Length") or 0)
            if content_length and content_length > MAX_STICKER_UPLOAD_BYTES:
                app.logger.info("Sticker candidate too large via header size=%s url=%s", content_length, sticker_url)
                continue
            payload = media.content
            if len(payload) > MAX_STICKER_UPLOAD_BYTES:
                app.logger.info("Sticker candidate too large via body size=%s url=%s", len(payload), sticker_url)
                continue
            image_key = upload_lark_image(payload)
            if not image_key:
                app.logger.warning("Lark image upload failed for sticker_url=%s", sticker_url)
                continue
            msg_id = send_lark_image_reply(reply_to_message_id, image_key)
            remember_bot_message(msg_id, chat_id)
            save_bot_text(chat_id, f"[sticker] {query} -> {sticker_query}", None, reply_to_message_id, event_message_id=msg_id)
            app.logger.info("Sticker reply sent for query=%s rewritten=%s candidate=%s", query, sticker_query, idx)
            return
        except Exception:
            app.logger.exception("Sticker reply candidate failed url=%s", sticker_url)
            continue

    app.logger.info("All sticker candidates failed for query=%s rewritten=%s", query, sticker_query)
    return


def should_respond(message: dict, content_obj: dict, normalized_text: str) -> tuple[bool, list[str]]:
    mentions = message.get("mentions") or content_obj.get("mentions") or []
    raw_text = content_obj.get("text") or ""
    segments = extract_bot_segments(raw_text, mentions)

    if segments:
        return True, segments
    if mentions and re.search(r"\bsticksy\b", (raw_text or normalized_text or ""), re.IGNORECASE):
        fallback = sanitize_text(normalized_text or raw_text)
        return (True, [fallback] if fallback else [])
    return False, []


def infer_language_hint(query: str) -> str:
    return "zh" if looks_like_cjk(query) else "same_as_user"


def generate_topics(chat_id: str) -> list[dict]:
    cutoff = now_ts() - TOPIC_WINDOW_HOURS * 3600
    rows = db_query_all(
        "SELECT sender_name, text_content FROM messages WHERE chat_id = :chat_id AND message_type = 'text' AND is_from_bot = 0 AND created_at_ts >= :cutoff ORDER BY created_at_ts DESC LIMIT 150",
        {"chat_id": chat_id, "cutoff": cutoff},
    )

    if not rows:
        return []

    lines = []
    for row in reversed(rows):
        name = row["sender_name"] or "Unknown"
        text = (row["text_content"] or "").strip()
        if text:
            lines.append(f"{name}: {text}")

    if not lines:
        return []

    cached = db_query_one(
        "SELECT topics_json, computed_at_ts FROM topic_cache WHERE chat_id = :chat_id",
        {"chat_id": chat_id},
    )
    if cached and now_ts() - int(cached["computed_at_ts"]) < 600:
        try:
            return json.loads(cached["topics_json"])
        except Exception:
            pass

    try:
        config = types.GenerateContentConfig(
            system_instruction=TOPIC_SYSTEM_PROMPT,
            response_mime_type="application/json",
        )
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=json.dumps({"messages": lines}, ensure_ascii=False),
            config=config,
        )
        parsed = json.loads((resp.text or "{}").strip())
        topics = parsed.get("topics") or []
        normalized = []
        for t in topics:
            topic = (t.get("topic") or "").strip()
            importance = int(t.get("importance") or 3)
            if topic:
                normalized.append({"topic": topic, "importance": max(1, min(5, importance))})

        if IS_POSTGRES:
            db_execute(
                """
                INSERT INTO topic_cache(chat_id, topics_json, computed_at_ts)
                VALUES(:chat_id, :topics_json, :computed_at_ts)
                ON CONFLICT (chat_id) DO UPDATE
                SET topics_json = EXCLUDED.topics_json, computed_at_ts = EXCLUDED.computed_at_ts
                """,
                {"chat_id": chat_id, "topics_json": json.dumps(normalized, ensure_ascii=False), "computed_at_ts": now_ts()},
            )
        else:
            db_execute(
                "INSERT OR REPLACE INTO topic_cache(chat_id, topics_json, computed_at_ts) VALUES(:chat_id, :topics_json, :computed_at_ts)",
                {"chat_id": chat_id, "topics_json": json.dumps(normalized, ensure_ascii=False), "computed_at_ts": now_ts()},
            )
        return normalized
    except Exception:
        return []


def resolve_chat_name(chat_id: str, fallback_name: str = "") -> str:
    if fallback_name and fallback_name.strip():
        return fallback_name.strip()

    row = db_query_one(
        "SELECT chat_name, cached_at_ts FROM chat_profile_cache WHERE chat_id = :chat_id",
        {"chat_id": chat_id},
    )
    if row and now_ts() - int(row["cached_at_ts"] or 0) < 7 * 86400 and (row.get("chat_name") or "").strip():
        return row["chat_name"].strip()

    chat_name = ""
    try:
        resp = requests.get(
            f"https://open.larkoffice.com/open-apis/im/v1/chats/{chat_id}",
            headers=lark_headers(),
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code < 400:
            body = resp.json()
            data = body.get("data") or {}
            item = data.get("item") or data.get("chat") or data
            chat_name = (item.get("name") or item.get("chat_name") or "").strip()
    except Exception:
        app.logger.exception("Failed to resolve chat name for chat_id=%s", chat_id)

    if chat_name:
        if IS_POSTGRES:
            db_execute(
                """
                INSERT INTO chat_profile_cache(chat_id, chat_name, cached_at_ts)
                VALUES(:chat_id, :chat_name, :cached_at_ts)
                ON CONFLICT (chat_id) DO UPDATE
                SET chat_name = EXCLUDED.chat_name, cached_at_ts = EXCLUDED.cached_at_ts
                """,
                {"chat_id": chat_id, "chat_name": chat_name, "cached_at_ts": now_ts()},
            )
        else:
            db_execute(
                "INSERT OR REPLACE INTO chat_profile_cache(chat_id, chat_name, cached_at_ts) VALUES(:chat_id, :chat_name, :cached_at_ts)",
                {"chat_id": chat_id, "chat_name": chat_name, "cached_at_ts": now_ts()},
            )
        return chat_name

    return chat_id


def recent_monitor_messages(chat_id: str, limit: int = 60) -> list[dict]:
    rows = db_query_all(
        """
        SELECT sender_name, text_content, created_at_ts
        FROM messages
        WHERE chat_id = :chat_id
          AND is_from_bot = 0
          AND message_type = 'text'
          AND text_content IS NOT NULL
          AND text_content <> ''
        ORDER BY created_at_ts DESC
        LIMIT :limit_rows
        """,
        {"chat_id": chat_id, "limit_rows": limit},
    )
    out = []
    for row in reversed(rows):
        out.append(
            {
                "sender_name": (row.get("sender_name") or "Unknown").strip() or "Unknown",
                "text": (row.get("text_content") or "").strip(),
                "ts": int(row.get("created_at_ts") or 0),
            }
        )
    return out


def build_topic_details(topics: list[dict], messages: list[dict]) -> list[dict]:
    if not topics:
        return []

    details = []
    for topic_obj in topics:
        topic = (topic_obj.get("topic") or "").strip()
        if not topic:
            continue
        terms = [t for t in re.findall(r"[\w\u4e00-\u9fff]+", topic.lower()) if len(t) > 1]
        matched = []
        for msg in messages:
            text_l = (msg.get("text") or "").lower()
            if terms and not any(t in text_l for t in terms):
                continue
            matched.append(msg)
            if len(matched) >= 8:
                break
        if not matched:
            matched = messages[-5:]
        details.append(
            {
                "topic": topic,
                "importance": int(topic_obj.get("importance") or 3),
                "messages": matched,
            }
        )
    return details


# -------- Web Routes --------


@app.route("/", methods=["POST"])
def webhook():
    maybe_sweep_retention()

    raw_body = request.get_data() or b""
    payload = request.get_json(silent=True) or {}

    valid, reason = verify_lark_request(raw_body, payload)
    if not valid:
        app.logger.warning("Webhook rejected: %s", reason)
        return Response(f"Unauthorized: {reason}", status=401)

    if "challenge" in payload:
        return jsonify({"challenge": payload["challenge"]})

    header = payload.get("header", {})
    event_type = header.get("event_type")
    if event_type and event_type != "im.message.receive_v1":
        app.logger.info("Ignoring event_type=%s", event_type)
        return "", 200

    event = payload.get("event", {})
    sender = event.get("sender") or {}
    if sender.get("sender_type") != "user":
        app.logger.info("Ignoring non-user sender_type=%s", sender.get("sender_type"))
        return "", 200

    message = event.get("message") or {}
    message_id = message.get("message_id")
    chat_id = message.get("chat_id")
    if not message_id or not chat_id:
        app.logger.warning("Missing message_id/chat_id in payload")
        return "", 200

    dedupe = (
        (f"msg:{message_id}" if message_id else None)
        or (f"evt:{header.get('event_id')}" if header.get("event_id") else None)
    )
    if dedupe and not record_event_once(dedupe):
        app.logger.info("Duplicate event ignored: %s", dedupe)
        return "", 200

    content_obj = {}
    try:
        content_obj = json.loads(message.get("content", "{}"))
    except Exception:
        content_obj = {}

    sender_open_id = ((sender.get("sender_id") or {}).get("open_id")) or ""
    sender_name = sender.get("sender_name") or sender.get("name") or ""
    profile = get_user_profile(sender_open_id)
    if not sender_name or sender_name.lower() == "unknown":
        sender_name = profile.get("display_name") or "Unknown"
    chat_name = event.get("chat_name") or ""
    msg_type = message.get("message_type") or ""
    text_content = extract_text_content(msg_type, content_obj)
    image_key = (content_obj.get("image_key") or "") if msg_type == "image" else ""
    mentions = message.get("mentions") or content_obj.get("mentions") or []
    if not isinstance(mentions, list):
        mentions = []

    create_time = int(message.get("create_time") or int(time.time() * 1000))
    created_at_ts = int(create_time / 1000)

    root_id = message.get("root_id") or message_id
    parent_id = message.get("parent_id") or ""
    app.logger.info(
        "Incoming message: chat_id=%s message_id=%s type=%s sender=%s root_id=%s parent_id=%s",
        chat_id,
        message_id,
        msg_type,
        sender_name,
        root_id,
        parent_id,
    )

    # Persist every incoming text-like/image message for future summarization.
    persist_type = "image" if msg_type == "image" else "text"
    if image_key or text_content:
        save_incoming_message(
            event_message_id=message_id,
            chat_id=chat_id,
            chat_name=chat_name,
            sender_open_id=sender_open_id,
            sender_name=sender_name,
            message_type=persist_type,
            text_content=text_content,
            image_key=image_key,
            root_id=root_id,
            parent_id=parent_id,
            created_at_ts=created_at_ts,
        )

    remember_non_bot_mentions(chat_id, mentions)

    should, segments = should_respond(message, content_obj, text_content)
    if not should:
        app.logger.info("No trigger detected for message_id=%s", message_id)
        return "", 200

    # If reply-triggered but no text, nothing actionable.
    if not segments:
        app.logger.info("Triggered but no actionable segments for message_id=%s", message_id)
        return "", 200

    user_tz = profile.get("tz_name") or get_user_timezone(sender_open_id)
    app.logger.info("Trigger accepted: segments=%s user_tz=%s", len(segments), user_tz)

    for seg in segments:
        if not seg:
            continue

        edit_instruction = parse_summary_edit_instruction(seg)
        if edit_instruction:
            try:
                original_summary, edited_summary = edit_latest_summary(chat_id, root_id, edit_instruction)
                if not original_summary:
                    edit_reply = "I couldn't find a recent summary to edit yet."
                elif not edited_summary:
                    edit_reply = "I couldn't apply that summary edit. Try a more specific edit instruction."
                else:
                    edit_reply = edited_summary
                msg_id = send_lark_text_reply(
                    message_id,
                    edit_reply,
                    mention_open_id=sender_open_id,
                    mention_name=sender_name,
                )
                remember_bot_message(msg_id, chat_id)
                save_bot_text(chat_id, edit_reply, root_id, message_id, event_message_id=msg_id)
            except Exception:
                app.logger.exception("Failed to process summary edit")
            continue
        if re.match(r"^\s*edit\s+summary\b", seg, re.IGNORECASE):
            try:
                prompt_text = 'Tell me what to edit after "Edit summary:".'
                msg_id = send_lark_text_reply(
                    message_id,
                    prompt_text,
                    mention_open_id=sender_open_id,
                    mention_name=sender_name,
                )
                remember_bot_message(msg_id, chat_id)
                save_bot_text(chat_id, prompt_text, root_id, message_id, event_message_id=msg_id)
            except Exception:
                app.logger.exception("Failed to send edit-summary prompt")
            continue

        learning_text = parse_learning_instruction(seg)
        if learning_text:
            try:
                save_learned_term(chat_id, learning_text, sender_name)
                rewritten_summary = rewrite_latest_summary_after_learning(chat_id, root_id, learning_text)
                learned_reply = rewritten_summary or f"Learned for this chat: {learning_text}"
                if rewritten_summary:
                    app.logger.info("Resent rewritten summary after learning update: chat_id=%s", chat_id)
                msg_id = send_lark_text_reply(
                    message_id,
                    learned_reply,
                    mention_open_id=sender_open_id,
                    mention_name=sender_name,
                )
                remember_bot_message(msg_id, chat_id)
                save_bot_text(chat_id, learned_reply, root_id, message_id, event_message_id=msg_id)
            except Exception:
                app.logger.exception("Failed to store learned term")
            continue
        if re.match(r"^\s*learn\b", seg, re.IGNORECASE):
            try:
                prompt_text = 'Tell me what to learn after "Learn".'
                msg_id = send_lark_text_reply(
                    message_id,
                    prompt_text,
                    mention_open_id=sender_open_id,
                    mention_name=sender_name,
                )
                remember_bot_message(msg_id, chat_id)
                save_bot_text(chat_id, prompt_text, root_id, message_id, event_message_id=msg_id)
            except Exception:
                app.logger.exception("Failed to send learn prompt")
            continue

        meeting_mode = meeting_request_mode(seg)
        if meeting_mode:
            meeting_url = choose_meeting_url_for_request(chat_id, seg, sender_open_id, sender_name)
            if not meeting_url:
                try:
                    missing_meeting = "I couldn't find a recent meeting transcript link in this chat yet."
                    msg_id = send_lark_text_reply(
                        message_id,
                        missing_meeting,
                        mention_open_id=sender_open_id,
                        mention_name=sender_name,
                    )
                    remember_bot_message(msg_id, chat_id)
                    save_bot_text(chat_id, missing_meeting, root_id, message_id, event_message_id=msg_id)
                except Exception:
                    app.logger.exception("Failed to send missing-meeting-link reply")
                continue

            try:
                minute_token = extract_minutes_token(meeting_url)
                cached = load_cached_meeting_summary(minute_token, meeting_mode) if minute_token else None
                if cached and (cached.get("reply_text") or "").strip():
                    meeting_reply = (cached.get("reply_text") or "").strip()
                    meeting_title = (cached.get("title") or "").strip() or "Meeting"
                    app.logger.info(
                        "Meeting summary cache hit: token=%s mode=%s source_chat=%s",
                        minute_token,
                        meeting_mode,
                        cached.get("source_chat_id") or "",
                    )
                else:
                    meeting_reply, meeting_title = build_meeting_reply(chat_id, seg, meeting_url, meeting_mode)
                    if minute_token:
                        save_cached_meeting_summary(
                            minute_token,
                            meeting_mode,
                            meeting_reply,
                            meeting_title,
                            source_chat_id=chat_id,
                        )
                        app.logger.info("Meeting summary cache stored: token=%s mode=%s", minute_token, meeting_mode)
                msg_id = send_lark_post_reply(
                    message_id,
                    meeting_reply,
                    title=meeting_title,
                )
                remember_bot_message(msg_id, chat_id)
                save_bot_text(chat_id, meeting_reply, root_id, message_id, event_message_id=msg_id)
            except MeetingTranscriptAccessError:
                app.logger.warning("Meeting transcript unavailable for url=%s", meeting_url)
                try:
                    unavailable_text = "I couldn't access that meeting transcript yet."
                    msg_id = send_lark_text_reply(
                        message_id,
                        unavailable_text,
                        mention_open_id=sender_open_id,
                        mention_name=sender_name,
                    )
                    remember_bot_message(msg_id, chat_id)
                    save_bot_text(chat_id, unavailable_text, root_id, message_id, event_message_id=msg_id)
                except Exception:
                    app.logger.exception("Failed to send meeting-unavailable reply")
                continue
            except Exception:
                app.logger.exception("Meeting summary/send failed")
                try:
                    unavailable_text = "I couldn't summarize that meeting just now."
                    msg_id = send_lark_text_reply(
                        message_id,
                        unavailable_text,
                        mention_open_id=sender_open_id,
                        mention_name=sender_name,
                    )
                    remember_bot_message(msg_id, chat_id)
                    save_bot_text(chat_id, unavailable_text, root_id, message_id, event_message_id=msg_id)
                except Exception:
                    app.logger.exception("Failed to send meeting-unavailable reply")
                continue

        elif looks_like_summary_request(seg):
            window = parse_time_window(seg, user_tz)
            rows = load_messages_for_window(chat_id, window.start_ts, window.end_ts, root_id if message.get("root_id") else None)
            if not rows:
                try:
                    send_missing_history_message(message_id, chat_id)
                    app.logger.info("No history found for summary window; sent fallback message")
                except Exception:
                    app.logger.exception("Failed to send missing-history fallback")
                    pass
                continue

            try:
                summary_text, image_keys = create_summary(
                    chat_id=chat_id,
                    query=seg,
                    asker_name=sender_name,
                    language_hint=infer_language_hint(seg),
                    rows=rows,
                    window_label=window.label,
                )

                msg_id = send_lark_text_reply(
                    message_id,
                    summary_text,
                    mention_open_id=sender_open_id,
                    mention_name=sender_name,
                )
                remember_bot_message(msg_id, chat_id)
                save_bot_text(chat_id, summary_text, root_id, message_id, event_message_id=msg_id)

                for key in image_keys[:MAX_SUMMARY_IMAGES]:
                    try:
                        image_msg_id = send_lark_image_reply(message_id, key)
                        remember_bot_message(image_msg_id, chat_id)
                    except Exception:
                        app.logger.exception("Failed sending summary image reply")
                        continue
            except Exception:
                # Outage/rate-limit behavior for summary: no message.
                app.logger.exception("Summary generation/send failed; intentionally silent to user")
                continue
        else:
            # Non-summary mention must reply with sticker only; on failure, stay silent.
            send_sticker_reply(message_id, chat_id, seg, current_message_id=message_id)
            app.logger.info("Processed sticker flow for segment")

    return "", 200


@app.route("/monitor", methods=["GET"])
def monitor_page():
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Sticksy Monitor</title>
  <style>
    :root {
      --bg: #000000;
      --bg2: #0b0b0b;
      --card: #121826;
      --ink: #f1f5ff;
      --muted: #9ba6c0;
      --accent: #36d6b0;
      --edge: #242d40;
    }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    .wrap { max-width: 1060px; margin: 0 auto; padding: 24px; }
    .title { font-size: 28px; font-weight: 700; margin-bottom: 6px; }
    .sub { color: var(--muted); margin-bottom: 20px; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(310px, 1fr));
      gap: 14px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--edge);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 10px 22px rgba(0,0,0,0.28);
    }
    .chat { font-size: 15px; font-weight: 700; margin-bottom: 4px; }
    .meta { font-size: 12px; color: var(--muted); margin-bottom: 10px; }
    ul { margin: 8px 0 0 18px; padding: 0; }
    li { margin: 4px 0; }
    .pill {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      background: #18373a;
      color: var(--accent);
      font-size: 11px;
      margin-right: 6px;
    }
    details.topic {
      margin-top: 8px;
      border: 1px solid var(--edge);
      border-radius: 10px;
      background: var(--bg2);
      padding: 8px 10px;
    }
    details.topic summary {
      cursor: pointer;
      color: #d8e2ff;
      font-weight: 600;
      list-style: none;
    }
    details.topic summary::-webkit-details-marker { display: none; }
    .msg {
      margin-top: 8px;
      font-size: 12px;
      color: #d1dbef;
      border-left: 2px solid #2f3b55;
      padding-left: 8px;
    }
    .sender {
      color: var(--accent);
      font-weight: 600;
      margin-right: 6px;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="title">Sticksy Monitor</div>
    <div class="sub">Live group activity with current discussion topics (auto-refresh every 30s)</div>
    <div id="grid" class="grid"></div>
  </div>

  <script>
    async function load() {
      const res = await fetch('/monitor/api/groups');
      const data = await res.json();
      const grid = document.getElementById('grid');
      grid.innerHTML = '';

      for (const g of data.groups) {
        const card = document.createElement('div');
        card.className = 'card';

        const topicItems = (g.topic_details || []).map(t => {
          const messages = (t.messages || []).map(m => {
            return `<div class="msg"><span class="sender">${m.sender_name}</span>${m.text}</div>`;
          }).join('');
          return `
            <details class="topic">
              <summary>${t.topic}</summary>
              ${messages || '<div class="msg">No recent matching messages.</div>'}
            </details>
          `;
        }).join('') || '<div class="msg">No active topics</div>';

        card.innerHTML = `
          <div class="chat">${g.chat_name}</div>
          <div class="meta">${g.chat_id}</div>
          <div>
            <span class="pill">1h msgs: ${g.msg_count_1h}</span>
            <span class="pill">24h msgs: ${g.msg_count_24h}</span>
            <span class="pill">Users: ${g.active_users_24h}</span>
          </div>
          <div>${topicItems}</div>
        `;
        grid.appendChild(card);
      }
    }

    load();
    setInterval(load, 30000);
  </script>
</body>
</html>
    """.strip()
    return Response(html, mimetype="text/html")


@app.route("/monitor/api/groups", methods=["GET"])
def monitor_groups():
    one_hour = now_ts() - 3600
    one_day = now_ts() - 24 * 3600
    groups = db_query_all(
        """
        SELECT chat_id, MAX(chat_name) AS chat_name,
               SUM(CASE WHEN created_at_ts >= :one_hour THEN 1 ELSE 0 END) AS msg_count_1h,
               SUM(CASE WHEN created_at_ts >= :one_day THEN 1 ELSE 0 END) AS msg_count_24h,
               COUNT(DISTINCT CASE WHEN created_at_ts >= :one_day THEN sender_open_id ELSE NULL END) AS active_users_24h
        FROM messages
        WHERE is_from_bot = 0
        GROUP BY chat_id
        ORDER BY msg_count_24h DESC
        LIMIT 50
        """,
        {"one_hour": one_hour, "one_day": one_day},
    )

    output = []
    for row in groups:
        chat_id = row["chat_id"]
        topics = generate_topics(chat_id)
        recent_messages = recent_monitor_messages(chat_id, limit=60)
        topic_details = build_topic_details(topics, recent_messages)
        resolved_name = resolve_chat_name(chat_id, fallback_name=(row.get("chat_name") or ""))
        output.append(
            {
                "chat_id": chat_id,
                "chat_name": resolved_name,
                "msg_count_1h": int(row["msg_count_1h"] or 0),
                "msg_count_24h": int(row["msg_count_24h"] or 0),
                "active_users_24h": int(row["active_users_24h"] or 0),
                "topics": topics,
                "topic_details": topic_details,
            }
        )

    return jsonify({"groups": output, "retention_days": RETENTION_DAYS})


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"ok": True, "model": GEMINI_MODEL, "retention_days": RETENTION_DAYS})


@app.route("/admin/clear-history", methods=["POST"])
def admin_clear_history():
    if not ADMIN_API_KEY:
        return jsonify({"ok": False, "error": "ADMIN_API_KEY is not configured"}), 403

    provided = request.headers.get("X-Admin-Key", "")
    if provided != ADMIN_API_KEY:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

    body = request.get_json(silent=True) or {}
    chat_id = (body.get("chat_id") or "").strip()

    if chat_id:
        db_execute("DELETE FROM messages WHERE chat_id = :chat_id", {"chat_id": chat_id})
        db_execute("DELETE FROM bot_messages WHERE chat_id = :chat_id", {"chat_id": chat_id})
        db_execute("DELETE FROM topic_cache WHERE chat_id = :chat_id", {"chat_id": chat_id})
        db_execute("DELETE FROM learned_terms WHERE chat_id = :chat_id", {"chat_id": chat_id})
        db_execute("DELETE FROM mentioned_identities WHERE chat_id = :chat_id", {"chat_id": chat_id})
        db_execute("DELETE FROM meeting_summary_cache WHERE source_chat_id = :chat_id", {"chat_id": chat_id})
        app.logger.warning("Admin cleared history for chat_id=%s", chat_id)
        return jsonify({"ok": True, "scope": "chat", "chat_id": chat_id})

    db_execute("DELETE FROM messages", {})
    db_execute("DELETE FROM bot_messages", {})
    db_execute("DELETE FROM processed_events", {})
    db_execute("DELETE FROM topic_cache", {})
    db_execute("DELETE FROM user_timezone_cache", {})
    db_execute("DELETE FROM user_profile_cache", {})
    db_execute("DELETE FROM learned_terms", {})
    db_execute("DELETE FROM mentioned_identities", {})
    db_execute("DELETE FROM meeting_summary_cache", {})
    app.logger.warning("Admin cleared all history")
    return jsonify({"ok": True, "scope": "all"})


@app.route("/debug/echo", methods=["GET", "POST"])
def debug_echo():
    body = request.get_json(silent=True)
    app.logger.info("debug_echo called method=%s body=%s", request.method, body)
    return jsonify({"ok": True, "method": request.method, "body": body})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
