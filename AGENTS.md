# AGENTS.md

- Tài liệu và text hiển thị cho người dùng viết tiếng Việt; code và commit message viết tiếng Anh.
- Không lưu access token, raw evidence hoặc secret trong conversation/event log.
- Tool write luôn cần proposal hash và xác nhận một lần từ user; không tự retry tool write.
- Mọi request phải có `org_id`, `user_id` và permissions từ signed Titan context.
- Capability chỉ gọi qua Control Plane gateway; không gọi `fm-mcp` hay warehouse trực tiếp.
