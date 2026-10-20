# Titan Agent Orchestrator

Service quản lý hội thoại, run, proposal, xác nhận và capability execution cho
Titan. Portal không gọi service này trực tiếp: Titan Control Plane xác thực user,
ký context rồi proxy request vào đây.

Repo này không sở hữu metric, dữ liệu nguồn, workflow FM hoặc MCP credential.
Nó gọi các capability do `titan-data-platform` công bố. Workflow/request/approval
vẫn thuộc `fm-mcp`.

## Run local

```bash
uv venv
uv pip install -e ".[dev]"
cp .env.example .env
fastapi dev src/titan_agent_orchestrator/main.py
```

## API flow

1. Control Plane tạo conversation với context user/tenant đã ký, có hạn dùng.
2. Gửi message tạo một run và event timeline.
3. Run có hành động ghi chuyển sang `waiting_confirmation`.
4. User quyết định; Control Plane chuyển quyết định, orchestrator mới gọi
   capability gateway.
5. Portal đọc event/result qua Control Plane.

Xem [architecture](docs/architecture.md) và [capability contract](docs/capability-contract.md).
