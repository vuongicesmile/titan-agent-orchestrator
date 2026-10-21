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
python -m uvicorn titan_agent_orchestrator.main:app --env-file .env --port 8010 --reload
```

## Docker Compose cho Orchestrator

Chạy tại repo này:

```bash
cp .env.example .env
# Điền hai secret dùng chung với Control Plane trong .env.
docker compose up -d --build --wait
docker compose ps
curl --fail http://localhost:8010/health
```

Service API tại `http://localhost:8010`, Swagger tại `http://localhost:8010/docs`.
`/health` kiểm tra tiến trình Orchestrator; kết nối Control Plane được sử dụng
khi gửi message và đọc capability catalog. Khi khai báo `LLM_BASE_URL`, `LLM_MODEL`
và `LLM_API_KEY`, planner dùng LLM theo chuẩn OpenAI-compatible để chọn capability.
Không có các biến này, service vẫn chạy với bộ định tuyến từ khóa tối giản để phục hồi an toàn.
Build image không cần API key LLM.

Planner chỉ chọn capability trong catalog đã được Control Plane lọc theo quyền. Nó không
cho LLM tự tạo input tool: hiện chỉ chuyển nguyên câu hỏi vào `data.answer_question`;
tool có tham số bắt buộc hoặc tool ghi vẫn phải có adapter được kiểm duyệt và quy trình xác nhận.

Để kết nối Control Plane local, cấu hình hai phía:

| Orchestrator `.env` | Control Plane `.env` |
| --- | --- |
| `TITAN_CAPABILITY_GATEWAY_TOKEN` | `TITAN_AGENT_GATEWAY_TOKEN` (cùng giá trị) |
| `TITAN_CONTEXT_SIGNING_SECRET` | `TITAN_AGENT_CONTEXT_SIGNING_SECRET` (cùng giá trị) |
| `TITAN_CONTROL_PLANE_URL=http://host.docker.internal:8000` | `TITAN_AGENT_ORCHESTRATOR_URL=http://127.0.0.1:8010` |

Control Plane trên host phải listen ở `0.0.0.0:8000` để container gọi được.
Nếu Control Plane cũng chạy trong Docker, dùng profile `agentic` bên dưới để
hai service gọi nhau qua tên service trên cùng network.

Conversation/run lưu trong volume `orchestrator_state`, vẫn còn khi recreate
container hoặc chạy `docker compose down`. Không thêm `--volumes` nếu cần giữ dữ liệu.
Image chạy bằng user không có quyền root; `.env` và dữ liệu local được loại khỏi
build context. Docker Desktop cần Linux engine đang chạy và bật WSL Integration
cho distro dùng để build.

## API flow

1. Control Plane tạo conversation với context user/tenant đã ký, có hạn dùng.
2. Gửi message tạo một run và event timeline.
3. Run có hành động ghi chuyển sang `waiting_confirmation`.
4. User quyết định; Control Plane chuyển quyết định, orchestrator mới gọi
   capability gateway.
5. Portal đọc event/result qua Control Plane.

Xem [architecture](docs/architecture.md) và [capability contract](docs/capability-contract.md).

## Chạy cùng Titan Data và FM MCP bằng Docker

Từ repo `titan-data-platform`, Docker Compose có profile `agentic`. Profile này
khởi động Titan Control Plane, Agent Orchestrator, FM Workflow API, FM MCP HTTP
và các dependency dữ liệu của Titan.

```bash
docker compose --profile agentic -f deploy/compose/docker-compose.yml up --build
```

Sau khi các healthcheck đều `healthy`:

1. Mở `http://localhost:8000` (Titan Data) và tạo/chọn tenant `demo`.
2. Vào **Cài đặt → Chẩn đoán MCP**, tạo connection từ profile `fm_mcp`, bật nó
   và chạy **Kiểm tra**. Registry nằm trong `TITAN_MCP_PROFILES`; credential
   chỉ ở biến môi trường của Control Plane.
3. Portal gọi `/api/tenants/{org}/orchestrator/...`; Control Plane ký context
   ngắn hạn và proxy sang service tại `http://localhost:8010`.

Trong môi trường chia sẻ, đặt giá trị ngẫu nhiên khác nhau cho
`TITAN_AGENT_CONTEXT_SIGNING_SECRET` và `TITAN_AGENT_GATEWAY_TOKEN`; không dùng
các giá trị development trong `.env.example`.
