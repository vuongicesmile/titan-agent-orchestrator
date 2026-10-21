# AGENTS.md

- Tài liệu viết tiếng Việt. Runtime message, error, event log, API-facing status text, code, identifier và LLM prompt viết tiếng Anh.
- Comment và docstring trong code viết tiếng Việt. Ngoại lệ: system prompt gửi tới LLM luôn viết tiếng Anh.
- Không lưu access token, raw evidence hoặc secret trong conversation/event log.
- Tool write luôn cần proposal hash và xác nhận một lần từ user; không tự retry tool write.
- Mọi request phải có `org_id`, `user_id` và permissions từ signed Titan context.
- Capability chỉ gọi qua Control Plane gateway; không gọi `fm-mcp` hay warehouse trực tiếp.
