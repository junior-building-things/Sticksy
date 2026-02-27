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
- The query should serve as a fun reply to the user's message (e.g. if they say "hi", the query can be "hi there", if they say "are you there", the query can be "hiding")
- If the user's message is factual/math/knowledge related, use playful uncertainty queries like "idk", "confused", or "shrug"
- Preserve the user's language when possible.
- Keep the query 1-4 words.
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
    _last_retention_sweep = current


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


def create_summary(query: str, asker_name: str, language_hint: str, rows: list[dict], window_label: str) -> tuple[str, list[str]]:
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


def build_sticker_search_keyword(user_query: str) -> str:
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
            contents=json.dumps({"user_query": raw}, ensure_ascii=False),
            config=config,
        )
        parsed = json.loads((resp.text or "{}").strip())
        keyword = sanitize_text(parsed.get("query") or parsed.get("keyword") or "")
        if keyword:
            return keyword[:80]
    except Exception:
        app.logger.exception("Sticker keyword rewrite failed; fallback to raw")
    return raw[:80]


def send_sticker_reply(reply_to_message_id: str, chat_id: str, query: str):
    sticker_query = build_sticker_search_keyword(query)
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

        if looks_like_summary_request(seg):
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
            send_sticker_reply(message_id, chat_id, seg)
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
      --bg: #f5f3ef;
      --card: #ffffff;
      --ink: #1e2a2f;
      --muted: #5d6a70;
      --accent: #0c7a6f;
      --edge: #d9d3c8;
    }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: radial-gradient(circle at 8% 8%, #fff3cd 0, var(--bg) 40%);
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
      box-shadow: 0 6px 18px rgba(40,40,40,0.06);
    }
    .chat { font-size: 15px; font-weight: 700; margin-bottom: 4px; }
    .meta { font-size: 12px; color: var(--muted); margin-bottom: 10px; }
    ul { margin: 8px 0 0 18px; padding: 0; }
    li { margin: 4px 0; }
    .pill {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      background: #e6f5f2;
      color: var(--accent);
      font-size: 11px;
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

        const topicItems = (g.topics || []).map(t => `<li>${t.topic}</li>`).join('') || '<li>No active topics</li>';

        card.innerHTML = `
          <div class="chat">${g.chat_name || g.chat_id}</div>
          <div class="meta">Chat ID: ${g.chat_id}</div>
          <div>
            <span class="pill">1h msgs: ${g.msg_count_1h}</span>
            <span class="pill">24h msgs: ${g.msg_count_24h}</span>
            <span class="pill">Users: ${g.active_users_24h}</span>
          </div>
          <ul>${topicItems}</ul>
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
        output.append(
            {
                "chat_id": chat_id,
                "chat_name": row["chat_name"],
                "msg_count_1h": int(row["msg_count_1h"] or 0),
                "msg_count_24h": int(row["msg_count_24h"] or 0),
                "active_users_24h": int(row["active_users_24h"] or 0),
                "topics": topics,
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
        app.logger.warning("Admin cleared history for chat_id=%s", chat_id)
        return jsonify({"ok": True, "scope": "chat", "chat_id": chat_id})

    db_execute("DELETE FROM messages", {})
    db_execute("DELETE FROM bot_messages", {})
    db_execute("DELETE FROM processed_events", {})
    db_execute("DELETE FROM topic_cache", {})
    db_execute("DELETE FROM user_timezone_cache", {})
    db_execute("DELETE FROM user_profile_cache", {})
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
