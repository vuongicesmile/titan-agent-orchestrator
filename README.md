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
khi gửi message và đọc capability catalog. Planner dùng Groq mặc định và chuyển
sang z.ai nếu Groq thất bại và đã cấu hình key dự phòng. Nếu không có provider
khả dụng, service dùng bộ định tuyến từ khóa tối giản để phục hồi an toàn.
Build image không cần API key LLM.

Planner chỉ chọn capability trong catalog đã được Control Plane lọc theo quyền. Nó không
cho LLM tự tạo input tool: hiện chỉ chuyển nguyên câu hỏi vào `data.answer_question`;
tool có tham số bắt buộc hoặc tool ghi vẫn phải có adapter được kiểm duyệt và quy trình xác nhận.

## Cấu hình LLM và dự phòng

Code LLM nằm trong `src/titan_agent_orchestrator/llm/`:

| File | Trách nhiệm |
| --- | --- |
| `models.py` | `ProviderConfig`, `LLMRequest`, `LLMResponse`, `TokenUsage`, `RouterDecision` |
| `router.py` | Chọn provider chính/dự phòng cho planner |
| `client.py` | HTTP transport, timeout và chuẩn hóa response |
| `exceptions.py` | Lỗi không chứa credential hoặc raw response |
| `providers/groq.py` | Groq mặc định, model `openai/gpt-oss-120b` |
| `providers/zai.py` | z.ai dự phòng, model `glm-4.7` |
| `providers/gemini.py` | Gemini qua API tương thích, model `gemini-2.5-flash` |

`orchestration/planner.py` chỉ chọn capability và kiểm tra input;
`adapters/titan.py` chỉ kết nối capability gateway của Control Plane.
Cấu trúc tham khảo cách ILMUchat tách scenario resolver, gateway client và schemas;
repo này dùng env, chưa thêm database/Redis cho cấu hình model.

Đặt trong `.env`:

```dotenv
LLM_PROVIDER=groq
LLM_FALLBACK_PROVIDER=zai
GROQ_API_KEY=<groq-key>
ZAI_API_KEY=<zai-key>
```

Mỗi provider có `<PROVIDER>_API_KEY`, `<PROVIDER>_BASE_URL`, `<PROVIDER>_MODEL`.
Endpoint và model có mặc định trong từng file provider; chỉ cần key để bật.
`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` cũ vẫn hoạt động cho provider chính;
biến riêng của provider được ưu tiên. Key chính không bao giờ tự chuyển sang
provider dự phòng. Không có `ZAI_API_KEY` thì bước dự phòng z.ai được bỏ qua.

Router thử mỗi provider tối đa một lần khi gặp timeout, lỗi HTTP, response rỗng,
bị cắt hoặc JSON sai schema. `LLM_TIMEOUT_S` áp dụng cho từng lần gọi; phía proxy
cần cho phép đủ thời gian cho cả chuỗi nếu bật dự phòng. Khi model trả quyết định
`null` hợp lệ, router không gọi thêm provider. Catalog validation và bộ luật của
planner vẫn áp dụng sau bước gọi LLM. Đây chỉ là dự phòng cho inference; việc
thực thi capability, xác nhận tool ghi và idempotency giữ nguyên.

Đặt `LLM_FALLBACK_PROVIDER=` để tắt dự phòng, hoặc `LLM_PROVIDER=gemini` để dùng
Gemini làm chính. Đổi env thì recreate container bằng `docker compose up -d`.
`LLMResponse` chuẩn hóa provider, model, content, finish reason và token usage;
`RouterDecision` chỉ chấp nhận trường `capability` dạng string hoặc null.

Tài liệu provider: [Groq](https://console.groq.com/docs/api-reference),
[z.ai](https://docs.z.ai/api-reference/llm/chat-completion),
[Gemini](https://ai.google.dev/gemini-api/docs/openai).

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

Portal có thể gửi message với `background: true` để nhận ngay run ở trạng
thái queued, sau đó đọc `/v1/runs/{id}` và `/events`. Mặc định vẫn chạy đồng
bộ để tương thích client cũ. Các endpoint quyết định giữ nguyên xác nhận,
không tự retry tool ghi; UI đọc tiến trình song song khi HTTP xác nhận đang
chờ. Background task hiện nằm trong process, chưa phải hàng đợi bền vững:
không restart khi có tool ghi chạy, không tự replay task sau restart.

`TITAN_CAPABILITY_TIMEOUT_S=180` tăng ngân sách chờ tool; kết nối vẫn 5 giây,
catalog 20 giây. Proxy Control Plane dùng `TITAN_AGENT_PROXY_TIMEOUT_S=300`.
Progress gồm các bước thực thi thật, không chứa suy luận nội bộ hoặc secret.

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
