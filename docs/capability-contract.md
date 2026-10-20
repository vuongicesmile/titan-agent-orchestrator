# Capability contract với Titan Control Plane

Orchestrator gọi một gateway nội bộ, không gọi MCP server trực tiếp. Gateway chỉ
chấp nhận service credential của orchestrator; context user có `expires_at` và
được Titan Control Plane ký HMAC trước khi chuyển vào service.

`GET /api/internal/agent/capabilities?org_id=<org_id>` trả catalog đã lọc theo
tenant và permission. Mỗi item gồm `name`, `description`, `input_schema`,
`write`, `requires_confirmation` và `timeout_seconds`.

`POST /api/internal/agent/capabilities/{name}/execute` nhận:

```json
{
  "org_id": "rmit_fm",
  "user_id": "user-123",
  "permissions": ["agent:execute"],
  "run_id": "...",
  "idempotency_key": "...",
  "arguments": {}
}
```

Gateway phải kiểm tra tenant, permission, schema, confirmation binding và audit.
Kết quả không được chứa credential; evidence lớn lưu tại Titan và chỉ trả reference.
