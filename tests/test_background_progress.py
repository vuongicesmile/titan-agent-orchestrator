import base64
import hashlib
import hmac
import json
import threading
from types import SimpleNamespace

import httpx
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

from titan_agent_orchestrator import auth
from titan_agent_orchestrator.adapters.titan import TitanCapabilityGateway
from titan_agent_orchestrator.main import create_app
from titan_agent_orchestrator.models import Principal
from titan_agent_orchestrator.persistence.sqlite import SQLiteRepository
from titan_agent_orchestrator.settings import Settings
from test_api import FakeGateway, signed_context


def test_background_receipt_progress_and_scope(monkeypatch, tmp_path):
    secret = "test-background-secret"
    monkeypatch.setattr(auth, "settings", Settings(context_signing_secret=secret))
    repository = SQLiteRepository(f"sqlite:///{tmp_path / 'background.db'}")
    entered, release = threading.Event(), threading.Event()

    class SlowGateway(FakeGateway):
        def catalog(self, principal):
            entered.set()
            assert release.wait(5)
            return super().catalog(principal)

    app = create_app(repository, SlowGateway())
    tasks = []
    monkeypatch.setattr(BackgroundTasks, "add_task", lambda self, func, *args, **kwargs: tasks.append((func, args)))
    client = TestClient(app)
    headers = {"X-Titan-Context": signed_context(secret)}
    cid = client.post('/v1/conversations', json={}, headers=headers).json()['id']
    response = client.post(f'/v1/conversations/{cid}/messages', json={
        'content': 'Show data trends', 'background': True,
    }, headers=headers)
    assert response.status_code == 202
    run = response.json()
    assert run['status'] == 'queued'
    assert not entered.is_set()
    func, args = tasks.pop()
    thread = threading.Thread(target=func, args=args)
    thread.start()
    try:
        assert entered.wait(2)
        current = client.get(f"/v1/runs/{run['id']}", headers=headers).json()
        assert current['status'] == 'running'
        events = client.get(f"/v1/runs/{run['id']}/events", headers=headers).json()
        assert events[-1]['message'] == 'Loading available capabilities.'
        assert repository.list_messages(cid)[0].role == 'user'
        token = signed_context(secret).split('.')[0]
        original = json.loads(base64.urlsafe_b64decode(token + '=' * (-len(token) % 4)))
        for override in ({'org_id': 'other_org'}, {'user_id': 'other-user'}):
            encoded = base64.urlsafe_b64encode(json.dumps({**original, **override}).encode()).decode().rstrip('=')
            signature = hmac.new(secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()
            other = {'X-Titan-Context': encoded + '.' + signature}
            assert client.get(f"/v1/runs/{run['id']}", headers=other).status_code == 404
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert client.get(f"/v1/runs/{run['id']}", headers=headers).json()['status'] == 'completed'


def test_background_failure_is_sanitized_and_not_retried(monkeypatch, tmp_path):
    secret = "test-background-secret"
    monkeypatch.setattr(auth, "settings", Settings(context_signing_secret=secret))
    app = create_app(SQLiteRepository(f"sqlite:///{tmp_path / 'failed.db'}"), FakeGateway())
    calls = []

    def broken(message, catalog):
        calls.append(message)
        raise RuntimeError('private-test-detail')

    app.state.engine.planner = SimpleNamespace(propose=broken)
    client = TestClient(app)
    headers = {"X-Titan-Context": signed_context(secret)}
    cid = client.post('/v1/conversations', json={}, headers=headers).json()['id']
    run = client.post(f'/v1/conversations/{cid}/messages', json={
        'content': 'test', 'background': True,
    }, headers=headers).json()
    result = client.get(f"/v1/runs/{run['id']}", headers=headers)
    assert result.json()['status'] == 'failed'
    assert 'private-test-detail' not in result.text
    assert len(calls) == 1


def test_gateway_has_long_tool_budget_but_short_catalog_timeout():
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=[] if request.method == 'GET' else {'answer': 'ok'})

    with httpx.Client(base_url='http://titan-api:8000', transport=httpx.MockTransport(handle)) as client:
        gateway = TitanCapabilityGateway('http://titan-api:8000', 'test', client=client, timeout_s=240)
        principal = Principal(org_id='demo', user_id='u')
        gateway.catalog(principal)
        gateway.execute(principal, 'run', 'key', 'reports.create_draft', {'prompt': 'test'})
    assert calls[0].extensions['timeout']['read'] == 20
    assert calls[1].extensions['timeout']['read'] == 240
    assert calls[1].extensions['timeout']['connect'] == 5
    assert len(calls) == 2
