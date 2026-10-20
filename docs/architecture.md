# Kiến trúc Titan Agent Orchestrator

```mermaid
flowchart LR
    Portal["Titan Portal"] --> CP["Titan Control Plane"]
    CP -->|"signed user context"| Agent["Agent Orchestrator"]
    Agent --> DB[("Conversation / run / event store")]
    Agent -->|"capability request"| CP
    CP --> QA["QA / Report / Studio"]
    CP --> MCP["MCP registry + policy"]
    MCP --> FM["fm-mcp"]
```

Orchestrator sở hữu lifecycle của hội thoại và run. Control Plane sở hữu
identity, tenant policy, credential và execution gateway. FM MCP sở hữu
nghiệp vụ workflow. Do đó model không có quyền trực tiếp gọi database, MCP
server hoặc service credential.

## State machine

```mermaid
stateDiagram-v2
    [*] --> queued
    queued --> running
    running --> waiting_confirmation: tool ghi được đề xuất
    running --> completed: chỉ đọc hoàn tất
    waiting_confirmation --> running: người dùng xác nhận
    waiting_confirmation --> cancelled: người dùng từ chối
    running --> completed: capability hoàn tất
    running --> failed: lỗi không thể tiếp tục
    queued --> cancelled
    waiting_confirmation --> cancelled
```
