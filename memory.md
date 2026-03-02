# Sticksy Memory

- Lark Minutes summarization requires the meeting notes link sharing range to be set to `Anyone with the link`. If the minutes link is not shared broadly enough, the app receives `403 permission deny` on the Minutes metadata/transcript/media APIs even when the app has the corresponding Minutes scopes.
- For meeting transcript summarization, publish and re-authorize the Lark app after changing Minutes scopes so the tenant token reflects the new permissions.
