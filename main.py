import json
import os
import re
import threading
import time
import hashlib
import hmac
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
- Never translate, localize, or romanize people's names. Preserve person names exactly as they appear in the transcript or known_people.
- If the transcript includes a speaker email, include it in owner_email.
- Use any provided learned terminology exactly when relevant.
- Always synthesize; do not copy long transcript lines verbatim.
- summary_bullets should be concise key takeaways with no labels.
- next_steps should capture owner-specific action items when the owner is clear.
- If an owner is unclear, leave owner_name empty rather than guessing.
- Keep next steps concrete and brief.
""".strip()


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
    return ((mention.get("id") or {}).get("open_id")) or ""


def is_bot_mention(mention: dict) -> bool:
    open_id = mention_open_id(mention)
    if LARK_BOT_OPEN_ID:
        if open_id == LARK_BOT_OPEN_ID:
            return True
        name = str(mention.get("name") or "").strip().lower()
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


def find_last_meeting_url(chat_id: str, limit: int = 80) -> str:
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
        {"chat_id": chat_id, "limit_rows": limit},
    )
    for row in rows:
        url = extract_minutes_url(row.get("text_content") or "")
        if url:
            return url
    return ""


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


def extract_participant_directory_from_obj(obj) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    name_keys = {"speaker", "speaker_name", "name", "user_name", "display_name", "participant_name", "attendee_name"}
    email_keys = {"email", "speaker_email", "user_email", "email_address"}

    def maybe_add(node):
        if not isinstance(node, dict):
            return

        participant_name = sanitize_text(first_nested_string(node, name_keys, max_depth=1))
        participant_email = first_nested_string(node, email_keys, max_depth=1).lower()
        participant_open_id = first_nested_id(node, max_depth=1)

        if not participant_name:
            return
        if not participant_email and not participant_open_id:
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


def fetch_lark_minute_transcript(minute_token: str) -> str:
    if not minute_token:
        return ""

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
        return ""

    if transcript_resp.status_code >= 400:
        app.logger.warning("Minutes transcript download failed: token=%s status=%s", minute_token, transcript_resp.status_code)
        return ""

    content_type = (transcript_resp.headers.get("Content-Type") or "").lower()
    if "application/json" not in content_type:
        text_payload = decode_response_text(transcript_resp)
        if len(text_payload) > MAX_MEETING_TRANSCRIPT_CHARS:
            text_payload = text_payload[:MAX_MEETING_TRANSCRIPT_CHARS]
        app.logger.info("Minutes transcript plain text length=%s token=%s", len(text_payload), minute_token)
        return text_payload

    try:
        body = transcript_resp.json()
    except Exception:
        app.logger.exception("Minutes transcript JSON parse failed: token=%s", minute_token)
        return ""

    if body.get("code", 0) not in {0, None}:
        app.logger.warning(
            "Minutes transcript api error: token=%s code=%s msg=%s",
            minute_token,
            body.get("code"),
            body.get("msg"),
        )
        return ""

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
        return direct_text[:MAX_MEETING_TRANSCRIPT_CHARS]

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
                return text_payload
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
    return extracted[:MAX_MEETING_TRANSCRIPT_CHARS] if extracted else ""


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


def lookup_owner_open_id(chat_id: str, owner_name: str) -> str:
    target = (owner_name or "").strip()
    if not target:
        return ""

    rows = db_query_all(
        """
        SELECT sender_open_id, sender_name
        FROM messages
        WHERE chat_id = :chat_id
          AND sender_open_id IS NOT NULL
          AND sender_open_id <> ''
          AND sender_name IS NOT NULL
          AND sender_name <> ''
        ORDER BY created_at_ts DESC
        LIMIT 200
        """,
        {"chat_id": chat_id},
    )
    for row in rows:
        sender_name = (row.get("sender_name") or "").strip()
        if person_name_matches(sender_name, target):
            return (row.get("sender_open_id") or "").strip()

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
    known_people: list[str],
    learned_terms: list[str],
) -> dict:
    user_payload = {
        "request": request_text,
        "minute_url": minute_url,
        "minute_meta": minute_meta,
        "transcript_excerpt": transcript_text,
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


def build_meeting_reply(chat_id: str, request_text: str, minute_url: str, mode: str) -> tuple[str, str]:
    minute_token = extract_minutes_token(minute_url)
    if not minute_token:
        raise RuntimeError("invalid meeting transcript link")

    minute_meta = {}
    try:
        minute_meta = fetch_lark_minute_meta(minute_token)
    except Exception:
        app.logger.exception("Minutes meta unavailable; continuing with transcript/media only")

    transcript_text = fetch_lark_minute_transcript(minute_token)
    if not transcript_text and minute_meta:
        transcript_text = extract_transcript_text_from_obj(minute_meta)
    media_bytes = None
    media_mime_type = None

    if len(transcript_text) < 800:
        media_bytes, media_mime_type = fetch_lark_minute_media(minute_token)

    if not transcript_text and not media_bytes:
        raise RuntimeError("meeting transcript is not accessible yet")

    known_people = known_people_for_chat(chat_id)
    transcript_directory = extract_transcript_speaker_emails(transcript_text)
    participant_directory = extract_participant_directory_from_obj(fetch_lark_minute_statistics(minute_token))
    speaker_directory = merge_people_directories(participant_directory, transcript_directory)
    app.logger.info(
        "Meeting people directory: token=%s transcript=%s participants=%s merged=%s",
        minute_token,
        len(transcript_directory),
        len(participant_directory),
        len(speaker_directory),
    )
    if not speaker_directory:
        app.logger.warning(
            "Meeting people directory empty: token=%s transcript=%s participants=%s",
            minute_token,
            len(transcript_directory),
            len(participant_directory),
        )
    for item in speaker_directory:
        speaker_name = (item.get("name") or "").strip()
        if not speaker_name:
            continue
        key = normalize_person_name(speaker_name)
        if key and all(normalize_person_name(existing) != key for existing in known_people):
            known_people.append(speaker_name)
    learned_terms = load_learned_terms(chat_id)
    parsed = analyze_meeting_with_gemini(
        request_text=request_text,
        minute_url=minute_url,
        minute_meta=minute_meta,
        transcript_text=transcript_text,
        media_bytes=media_bytes,
        media_mime_type=media_mime_type,
        known_people=known_people,
        learned_terms=learned_terms,
    )

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

        learning_text = parse_learning_instruction(seg)
        if learning_text:
            try:
                save_learned_term(chat_id, learning_text, sender_name)
                learned_reply = f"Learned for this chat: {learning_text}"
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
            meeting_url = extract_minutes_url(seg) or find_last_meeting_url(chat_id)
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
                meeting_reply, meeting_title = build_meeting_reply(chat_id, seg, meeting_url, meeting_mode)
                msg_id = send_lark_post_reply(
                    message_id,
                    meeting_reply,
                    title=meeting_title,
                )
                remember_bot_message(msg_id, chat_id)
                save_bot_text(chat_id, meeting_reply, root_id, message_id, event_message_id=msg_id)
            except Exception:
                app.logger.exception("Meeting summary/send failed")
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
        app.logger.warning("Admin cleared history for chat_id=%s", chat_id)
        return jsonify({"ok": True, "scope": "chat", "chat_id": chat_id})

    db_execute("DELETE FROM messages", {})
    db_execute("DELETE FROM bot_messages", {})
    db_execute("DELETE FROM processed_events", {})
    db_execute("DELETE FROM topic_cache", {})
    db_execute("DELETE FROM user_timezone_cache", {})
    db_execute("DELETE FROM user_profile_cache", {})
    db_execute("DELETE FROM learned_terms", {})
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
