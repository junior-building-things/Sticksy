# Sticksy Deployment (Render)

## 1. Prepare repo
- Push this folder to a GitHub repo.
- Keep secrets out of source control.

## 2. Create Render web service
- In Render, choose `New +` -> `Web Service`.
- Connect your repo.
- Render will read `render.yaml` automatically.

## 3. Set environment variables in Render
Copy from `.env.example` and fill real values:
- `GEMINI_API_KEY`
- `LARK_APP_ID`
- `LARK_APP_SECRET`
- `LARK_BOT_OPEN_ID` (`ou_32bdd3b66534e18bdd7bc490c075584e`)
- `KLIPY_API_KEY`
- `KLIPY_SEARCH_URL` (keep if correct)
- `DATABASE_URL` (managed Postgres URL from Neon/Supabase/Render Postgres)

Optional:
- `LARK_VERIFICATION_TOKEN`
- `LARK_REQUEST_SIGNING_SECRET`
- `THOMAS_OPEN_ID`
- `THOMAS_DISPLAY_NAME`
- `RETENTION_DAYS` (default `30`)
- `DATABASE_PATH` (local dev fallback only; ignored when `DATABASE_URL` is set)

## 4. Get your public webhook URL
After deploy, your base URL will look like:
- `https://<render-service>.onrender.com`

Webhook callback root for Lark:
- `https://<render-service>.onrender.com/`

Monitor dashboard:
- `https://<render-service>.onrender.com/monitor`

## 5. Configure Lark app
- Enable Event Subscription.
- Request URL: `https://<render-service>.onrender.com/`
- Subscribe event: `im.message.receive_v1`
- If you configured verify/signing secrets in Render, set the same values in Lark.

## 6. Verify health
- Open `https://<render-service>.onrender.com/healthz`
- Expect JSON with `"ok": true`

## 7. Quick behavior tests in Lark
- `@Sticksy Summarize`
- `@Sticksy Summarize yesterday's conversation`
- `@Sticksy Summarize last 2h conversation`
- `@Sticksy hi` (sticker reply)

## Notes
- If Klipy returns no sticker or API errors, Sticksy will stay silent by design.
- If Gemini/Lark errors occur, Sticksy will stay silent for summary/sticker failures by design.
- Free Render web services do not support persistent disks, so use `DATABASE_URL` for durable chat history.

## Clear history (admin)
Set `ADMIN_API_KEY` in Render, then call:

- Clear all history:
  `curl -X POST https://<render-service>.onrender.com/admin/clear-history -H "X-Admin-Key: <ADMIN_API_KEY>"`

- Clear one chat only:
  `curl -X POST https://<render-service>.onrender.com/admin/clear-history -H "Content-Type: application/json" -H "X-Admin-Key: <ADMIN_API_KEY>" -d '{"chat_id":"oc_xxx"}'`
