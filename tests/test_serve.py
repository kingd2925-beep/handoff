import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest
import torch

from handoff import serve


class StubAgent:
    def __init__(self, p_strong=0.8):
        self.p_strong = p_strong
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        return {"answers": {"hard": {"noul": self.p_strong}}}


def fake_embed(texts):
    """Encodes the number in the text as the first feature: 'x=-2' → [[-2.0, 0.0]]."""
    return [[float(t.split("=")[1]), 0.0] for t in texts]


def trained_head(gate=0.3, mean=(0.0, 0.0), std=(1.0, 1.0)):
    # p_strong = sigmoid((x0 - mean0) / std0)
    head = {"w": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0]),
            "mean": torch.tensor(mean), "std": torch.tensor(std)}
    if gate is not None:
        head["gate"] = gate
    return head


@pytest.fixture
def use_brain(monkeypatch):
    """Install a fresh Brain whose loader returns the given (agent, embed, head)."""
    def install(agent=None, embed=fake_embed, head=None):
        loads = []

        def loader():
            loads.append(1)
            return agent or StubAgent(), embed, head

        brain = serve.Brain(loader=loader)
        brain.loads = loads
        monkeypatch.setattr(serve, "BRAIN", brain)
        return brain
    return install


# ── require / handle ──────────────────────────────────────────────────
def test_require_returns_present_value():
    assert serve.require({"request": "hi"}, "request", str) == "hi"


@pytest.mark.parametrize("body", [{}, {"request": ""}, {"request": 5}, {"request": None}, {"request": ["x"]}])
def test_require_rejects_missing_empty_or_wrong_type(body):
    with pytest.raises(ValueError, match="'request' is missing or empty"):
        serve.require(body, "request", str)


def test_handle_unknown_path_raises_lookup_error():
    with pytest.raises(LookupError):
        serve.handle("/nope", {"request": "x"})


@pytest.mark.parametrize("path, body", [
    ("/route", {}),
    ("/route", {"request": ""}),
    ("/ask", {"questions": {"q": {}}}),
    ("/ask", {"state": "s", "questions": "not a dict"}),
    ("/ask", {"state": 42, "questions": {"q": {}}}),
])
def test_handle_bad_bodies_raise_value_error(path, body, use_brain):
    brain = use_brain()

    with pytest.raises(ValueError):
        serve.handle(path, body)
    assert brain.loads == []   # validation happens before the model is loaded


def test_handle_ask_returns_raw_answers(use_brain):
    agent = StubAgent(0.3)
    use_brain(agent=agent)
    questions = {"hard": {"type": "noul", "instructions": "?"}}

    out = serve.handle("/ask", {"state": {"request": "hello"}, "questions": questions})

    assert out == {"hard": {"noul": 0.3}}
    assert agent.calls == [({"request": "hello"}, questions)]


# ── route ─────────────────────────────────────────────────────────────
def test_route_without_trained_head_uses_zero_shot(use_brain):
    agent = StubAgent(0.8)
    use_brain(agent=agent, head=None)

    out = serve.handle("/route", {"request": "prove this theorem"})

    assert out == {"route": "strong", "p_strong": 0.8, "source": "zero-shot"}
    assert agent.calls[0][0] == {"request": "prove this theorem"}


def test_route_zero_shot_below_half_goes_cheap(use_brain):
    use_brain(agent=StubAgent(0.2), head=None)

    assert serve.route("say hi")["route"] == "cheap"


@pytest.mark.parametrize("text, expected", [("x=-2", "cheap"), ("x=0", "strong"), ("x=3", "strong")])
def test_route_with_trained_head_compares_p_to_gate(text, expected, use_brain):
    # sigmoid(-2)=0.119 < gate 0.3 → cheap; sigmoid(0)=0.5 → strong
    use_brain(head=trained_head(gate=0.3))

    out = serve.route(text)

    assert out["route"] == expected
    assert out["source"] == "trained"
    assert out["p_strong"] == round(torch.sigmoid(torch.tensor(float(text[2:]))).item(), 4)


def test_route_with_trained_head_normalises_with_mean_and_std(use_brain):
    use_brain(head=trained_head(gate=0.3, mean=(4.0, 0.0), std=(2.0, 1.0)))

    out = serve.route("x=0")   # (0 - 4) / 2 = -2 → p 0.1192

    assert out == {"route": "cheap", "p_strong": 0.1192, "source": "trained"}


def test_route_falls_back_to_default_gate_when_head_has_none(use_brain):
    use_brain(head=trained_head(gate=None))

    out = serve.route("x=-2")   # p 0.1192 >= DEFAULT_GATE 0.10

    assert serve.DEFAULT_GATE == 0.10
    assert out["route"] == "strong"


def test_route_with_zero_gate_never_hands_off(use_brain):
    use_brain(head=trained_head(gate=0.0))

    assert serve.route("x=-50")["route"] == "strong"


# ── Brain ─────────────────────────────────────────────────────────────
def test_brain_loads_once_and_unload_clears_state(use_brain):
    # Arrange
    brain = use_brain(head=trained_head())

    # Act
    brain.ensure()
    brain.ensure()
    loaded = (brain.agent, brain.embed, brain.head)
    brain.unload()

    # Assert
    assert len(brain.loads) == 1
    assert all(part is not None for part in loaded)
    assert brain.agent is None and brain.embed is None and brain.head is None


def test_brain_reloads_after_unload(use_brain):
    brain = use_brain()
    brain.ensure()
    brain.unload()

    brain.ensure()

    assert len(brain.loads) == 2 and brain.agent is not None


def test_brain_unload_when_not_loaded_is_a_no_op(use_brain):
    brain = use_brain()

    brain.unload()

    assert brain.agent is None and brain.loads == []


def test_brain_idle_grows_from_last_use(use_brain):
    brain = use_brain()
    brain.last_used -= 30

    assert brain.idle() >= 30
    brain.ensure()
    assert brain.idle() < 5


# ── HTTP handler on an ephemeral port ─────────────────────────────────
SECRET = "test-secret-token"


@pytest.fixture
def server(use_brain, monkeypatch):
    brain = use_brain(agent=StubAgent(0.8), head=None)
    monkeypatch.setattr(serve.Handler, "secret", SECRET)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    httpd.brain = brain
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def _headers(extra=None, token=SECRET, content_type="application/json"):
    headers = {}
    if token is not None:
        headers["X-Handoff-Token"] = token
    if content_type is not None:
        headers["Content-Type"] = content_type
    headers.update(extra or {})
    return headers


def _request(httpd, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
    try:
        conn.request(method, path, body=body, headers=_headers() if headers is None else headers)
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())
    finally:
        conn.close()


def _raw_post(httpd, content_length, headers=None):
    """POST with a hand-set Content-Length and no body."""
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
    try:
        conn.putrequest("POST", "/route")
        for key, value in {**_headers(), **(headers or {}), "Content-Length": content_length}.items():
            conn.putheader(key, value)
        conn.endheaders()
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())
    finally:
        conn.close()


def test_http_health_reports_state_and_proof(server):
    status, body = _request(server, "GET", "/health")

    assert status == 200
    assert body == {"loaded": False, "idle_s": 0, "trained": False, "proof": serve.proof(SECRET)}
    assert SECRET not in json.dumps(body)   # the token itself is never echoed


def test_proof_is_deterministic_and_secret_dependent():
    assert serve.proof("a") == serve.proof("a")
    assert serve.proof("a") != serve.proof("b")
    assert len(serve.proof("a")) == 16


def test_http_unknown_get_is_404(server):
    assert _request(server, "GET", "/secret")[0] == 404


def test_http_route_returns_decision_with_timing(server):
    status, body = _request(server, "POST", "/route", json.dumps({"request": "hard question"}))

    assert status == 200
    assert body["route"] == "strong" and body["source"] == "zero-shot"
    assert isinstance(body["ms"], int)


def test_http_accepts_localhost_host_and_json_charset(server):
    port = server.server_address[1]
    headers = _headers({"Host": f"localhost:{port}"}, content_type="application/json; charset=utf-8")

    status, _ = _request(server, "POST", "/route", b'{"request": "hi there"}', headers)

    assert status == 200


# token
@pytest.mark.parametrize("method, path, body", [("GET", "/health", None), ("POST", "/route", b'{"request": "x"}')])
@pytest.mark.parametrize("token", [None, "", "wrong-token", SECRET + "x"])
def test_http_missing_or_wrong_token_is_401(server, method, path, body, token):
    status, reply = _request(server, method, path, body, _headers(token=token))

    assert status == 401 and "X-Handoff-Token" in reply["error"]
    assert server.brain.loads == []   # refused before the model is touched


def test_http_non_ascii_token_is_401_not_a_crash(server):
    status, _ = _request(server, "GET", "/health", headers=_headers(token="café"))

    assert status == 401


def test_http_refuses_everything_when_server_has_no_secret(server, monkeypatch):
    monkeypatch.setattr(serve.Handler, "secret", None)

    assert _request(server, "GET", "/health", headers=_headers(token=""))[0] == 401


# host
@pytest.mark.parametrize("host", ["evil.com", "evil.com:{port}", "127.0.0.1", "127.0.0.1:1", "localhost",
                                  "attacker.example:{port}", "127.0.0.1.nip.io:{port}"])
def test_http_bad_host_is_403_dns_rebinding(server, host):
    port = server.server_address[1]

    status, reply = _request(server, "POST", "/route", b'{"request": "x"}',
                             _headers({"Host": host.format(port=port)}))

    assert status == 403 and "Host" in reply["error"]
    assert server.brain.loads == []


def test_http_bad_host_is_checked_before_token(server):
    status, _ = _request(server, "GET", "/health", headers=_headers({"Host": "evil.com"}, token=None))

    assert status == 403


# content type (a browser form / fetch "simple request" can't send application/json cross-site)
@pytest.mark.parametrize("content_type", [None, "text/plain", "application/x-www-form-urlencoded",
                                          "multipart/form-data"])
def test_http_non_json_content_type_is_415(server, content_type):
    status, reply = _request(server, "POST", "/route", b'{"request": "x"}', _headers(content_type=content_type))

    assert status == 415 and "Content-Type" in reply["error"]
    assert server.brain.loads == []


# body size / shape
def test_http_empty_body_is_413(server):
    status, body = _request(server, "POST", "/route", b"")

    assert status == 413 and "body must be" in body["error"]


def test_http_oversized_content_length_is_413(server):
    assert _raw_post(server, str(serve.MAX_BODY + 1))[0] == 413


def test_http_negative_content_length_is_413(server):
    assert _raw_post(server, "-5")[0] == 413


@pytest.mark.parametrize("value", ["abc", "12abc", "1.5"])
def test_http_unparseable_content_length_is_400(server, value):
    status, body = _raw_post(server, value)

    assert status == 400 and "Content-Length" in body["error"]


@pytest.mark.parametrize("raw", [b"{not json", b'["a list"]', b'{"request": ""}'])
def test_http_bad_bodies_are_400(server, raw):
    status, body = _request(server, "POST", "/route", raw)

    assert status == 400 and body["error"]


def test_http_unknown_post_path_is_404(server):
    assert _request(server, "POST", "/nope", b'{"a": 1}')[0] == 404


def test_http_loader_failure_is_500_not_a_crash(monkeypatch, server):
    def broken():
        raise RuntimeError("weights missing")
    monkeypatch.setattr(serve, "BRAIN", serve.Brain(loader=broken))

    status, body = _request(server, "POST", "/route", b'{"request": "hi there"}')

    assert status == 500 and "weights missing" in body["error"]


# ── _load: trained head goes through torch.load(weights_only=True) ───
def test_load_reads_a_trained_head_with_weights_only(monkeypatch, handoff_home):
    # Arrange
    from handoff import paths, train
    from conftest import REAL_SERVE_LOAD
    agent = StubAgent()
    monkeypatch.setattr(train, "load_laya", lambda: (agent, fake_embed))
    head = trained_head(gate=0.24)
    torch.save(head, paths.ensure(paths.HEAD))

    # Act
    loaded_agent, embed, loaded = REAL_SERVE_LOAD()

    # Assert
    assert loaded_agent is agent and embed is fake_embed
    assert loaded["gate"] == 0.24 and torch.equal(loaded["w"], head["w"])


def test_load_without_head_returns_none(monkeypatch):
    from handoff import train
    from conftest import REAL_SERVE_LOAD
    monkeypatch.setattr(train, "load_laya", lambda: (StubAgent(), fake_embed))

    assert REAL_SERVE_LOAD()[2] is None


class _Evil:
    def __reduce__(self):
        return (print, ("pwned",))


def test_load_refuses_a_pickled_object_in_the_head_file(monkeypatch):
    import pickle
    from handoff import paths, train
    from conftest import REAL_SERVE_LOAD
    monkeypatch.setattr(train, "load_laya", lambda: (StubAgent(), fake_embed))
    torch.save({"w": _Evil()}, paths.ensure(paths.HEAD))

    with pytest.raises(pickle.UnpicklingError):
        REAL_SERVE_LOAD()
