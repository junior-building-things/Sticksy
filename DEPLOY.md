# Sticksy Deployment

## Vercel

### 1. Create the project
- Push this folder to GitHub.
- In Vercel, import the repo and set the project root to `sticksy`.
- Vercel will use [`vercel.json`](/Users/thomas/Documents/vibe-coding/sticksy/vercel.json) and route all paths to [`api/index.py`](/Users/thomas/Documents/vibe-coding/sticksy/api/index.py).

### 2. Set environment variables in Vercel
- `GEMINI_API_KEY`
- `LARK_APP_ID`
- `LARK_APP_SECRET`
- `LARK_BOT_OPEN_ID`
- `DATABASE_URL` (recommended for durable history; Neon/Supabase/Render Postgres all work)

Optional:
- `LARK_VERIFICATION_TOKEN`
- `LARK_REQUEST_SIGNING_SECRET`
- `THOMAS_OPEN_ID`
- `THOMAS_DISPLAY_NAME`
- `RETENTION_DAYS` (default `30`)
- `DATABASE_PATH` (local/dev fallback only; ignored when `DATABASE_URL` is set)
- `MAX_ASSISTANT_CONTEXT_MESSAGES`
- `MAX_DOC_LOOKBACK_MESSAGES`
- `MAX_DOC_CONTENT_CHARS`
- `MAX_DOC_PROMPT_CHARS`
- `KLIPY_API_KEY`
- `KLIPY_SEARCH_URL`

### 3. Lark app scopes and access
- Enable Event Subscription and set the request URL to your Vercel deployment root, for example `https://your-bot.vercel.app/`.
- Subscribe to `im.message.receive_v1`.
- Grant the bot message reply permissions it already uses.
- Grant document read access for Lark Docs.
- Grant document comment access so the bot can add comments.
- If you want wiki links to work, make sure the app/bot can read the wiki node and is added where needed in the knowledge base.
- If you configure `LARK_VERIFICATION_TOKEN` or `LARK_REQUEST_SIGNING_SECRET`, set the same values in Lark.

### 4. Gemini through Vercel
- The bot reads `GEMINI_API_KEY` from the Vercel project environment, so Gemini calls run from your Vercel deployment.
- If you later want to move this to Vercel AI Gateway instead of a direct Gemini key, the assistant call sites are all centralized in [`main.py`](/Users/thomas/Documents/vibe-coding/sticksy/main.py).

### 5. Verify after deploy
- `https://<your-project>.vercel.app/healthz`
- `https://<your-project>.vercel.app/monitor`
- In Lark:
  - `@Sticksy what did we decide here?`
  - `@Sticksy summarize this doc https://.../docx/...`
  - `@Sticksy add a comment on this doc: Please tighten the success metric section`
  - `@Sticksy add a comment on "exact text from the doc": This claim needs a source`

## Render

### 1. Prepare repo
- Push this folder to a GitHub repo.
- Keep secrets out of source control.

### 2. Create Render web service
- In Render, choose `New +` -> `Web Service`.
- Connect your repo.
- Render will read `render.yaml` automatically.

### 3. Set environment variables in Render
Copy the same variables listed in the Vercel section above.

### 4. Get your public webhook URL
After deploy, your base URL will look like:
- `https://<render-service>.onrender.com`

Webhook callback root for Lark:
- `https://<render-service>.onrender.com/`

Monitor dashboard:
- `https://<render-service>.onrender.com/monitor`

### 5. Configure Lark app
- Enable Event Subscription.
- Request URL: `https://<render-service>.onrender.com/`
- Subscribe event: `im.message.receive_v1`
- If you configured verify/signing secrets in Render, set the same values in Lark.

### 6. Verify health
- Open `https://<render-service>.onrender.com/healthz`
- Expect JSON with `"ok": true`

### 7. Quick behavior tests in Lark
- `@Sticksy Summarize`
- `@Sticksy Summarize yesterday's conversation`
- `@Sticksy Summarize last 2h conversation`
- `@Sticksy summarize this doc https://.../docx/...`
- `@Sticksy add a comment on this doc: Please tighten the success metric section`

## Notes
- If Gemini or Lark fail during the assistant/doc flows, the bot now tries to send a short plain-language failure instead of silently falling back to stickers.
- Free Render web services do not support persistent disks, so use `DATABASE_URL` for durable chat history.
- Quoted doc comments are best-effort: the bot tries to anchor the comment to an exact excerpt first, then falls back to a whole-document comment if Lark rejects the quoted payload.

## Clear history (admin)
Set `ADMIN_API_KEY` in Render, then call:

- Clear all history:
  `curl -X POST https://<render-service>.onrender.com/admin/clear-history -H "X-Admin-Key: <ADMIN_API_KEY>"`

- Clear one chat only:
  `curl -X POST https://<render-service>.onrender.com/admin/clear-history -H "Content-Type: application/json" -H "X-Admin-Key: <ADMIN_API_KEY>" -d '{"chat_id":"oc_xxx"}'`
