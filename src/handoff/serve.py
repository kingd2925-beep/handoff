"""The local router service. 127.0.0.1 only; loads Laya on the first request, unloads it after
IDLE_UNLOAD seconds (hands the ~3 GB back on small Macs) and exits after EXIT_AFTER seconds.

  GET  /health                         → {"loaded", "idle_s", "trained", "proof"}
  POST /route {"request": "..."}        → {"route": "cheap" | "strong", "p_strong", "source", "ms"}
  POST /ask   {"state", "questions"}    → raw Laya answers (choice / score / noul)

Every request must carry the secret from ~/.handoff/token in an X-Handoff-Token header and a Host header
of 127.0.0.1/localhost, so web pages (CSRF, DNS rebinding) and other programs can't drive the service.
"""
import gc
import hashlib
import hmac
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import paths

HOST = "127.0.0.1"
PORT = int(os.environ.get("HANDOFF_PORT", "7071"))
IDLE_UNLOAD = int(os.environ.get("HANDOFF_IDLE", "600"))
EXIT_AFTER = int(os.environ.get("HANDOFF_EXIT", "1800"))
MAX_BODY = 64 * 1024
DEFAULT_GATE = 0.10


class Brain:
    def __init__(self, loader=None):
        self.lock = threading.Lock()
        self.agent = self.embed = self.head = None
        self.last_used = time.time()
        self._loader = loader

    def ensure(self):
        with self.lock:
            self.last_used = time.time()
            if self.agent is None:
                self.agent, self.embed, self.head = (self._loader or _load)()

    def unload(self):
        with self.lock:
            if self.agent is None:
                return
            self.agent = self.embed = self.head = None
            gc.collect()
            _free_gpu()

    def idle(self) -> float:
        return time.time() - self.last_used


def _load():
    from .train import load_laya
    import torch
    agent, embed = load_laya()
    head = None
    if paths.HEAD.exists():
        head = torch.load(paths.HEAD, map_location="cpu", weights_only=True)   # no pickle code execution
    return agent, embed, head


def _free_gpu():
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # freeing memory must never crash the server
        pass


BRAIN = Brain()
ZERO_SHOT_Q = {"hard": {"type": "noul", "instructions":
    "Does `request` need long multi-step reasoning, careful engineering, money math, legal judgement or specialist knowledge?"}}


def route(request: str) -> dict:
    BRAIN.ensure()
    head = BRAIN.head
    if head is None:
        p = float(BRAIN.agent.predict({"request": request}, ZERO_SHOT_Q)["answers"]["hard"]["noul"])
        return {"route": "strong" if p >= 0.5 else "cheap", "p_strong": round(p, 4), "source": "zero-shot"}
    import torch
    x = (torch.tensor(BRAIN.embed([request]), dtype=torch.float32) - head["mean"]) / head["std"]
    p = torch.sigmoid(x @ head["w"] + head["b"]).item()
    gate = head.get("gate", DEFAULT_GATE)
    return {"route": "cheap" if p < gate else "strong", "p_strong": round(p, 4), "source": "trained"}


def ask(state, questions: dict) -> dict:
    BRAIN.ensure()
    return BRAIN.agent.predict(state, questions)["answers"]


def require(body: dict, key: str, kinds) -> object:
    value = body.get(key)
    if not isinstance(value, kinds) or not value:
        raise ValueError(f"'{key}' is missing or empty")
    return value


def handle(path: str, body: dict) -> dict:
    if path == "/route":
        return route(require(body, "request", str))
    if path == "/ask":
        return ask(require(body, "state", (str, dict, list)), require(body, "questions", dict))
    raise LookupError(path)


def proof(secret: str) -> str:
    """Lets the CLI check it is talking to the real service without sending the token back."""
    return hashlib.sha256(("handoff:" + secret).encode()).hexdigest()[:16]


class Handler(BaseHTTPRequestHandler):
    secret = None   # set in main(); tests may set it directly

    def log_message(self, *args):
        return

    def _gate(self) -> bool:
        """Host + token check. Replies and returns False when the request must be refused."""
        port = self.server.server_address[1]
        if self.headers.get("Host", "") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            self.reply(403, {"error": "bad Host header"})
            return False
        sent = self.headers.get("X-Handoff-Token", "")
        if not (self.secret and hmac.compare_digest(sent.encode(), self.secret.encode())):   # bytes: non-ASCII can't crash
            self.reply(401, {"error": "missing or wrong X-Handoff-Token (see ~/.handoff/token)"})
            return False
        return True

    def reply(self, code: int, obj: dict):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self._gate():
            return
        if self.path != "/health":
            return self.reply(404, {"error": "not found"})
        self.reply(200, {"loaded": BRAIN.agent is not None, "idle_s": round(BRAIN.idle()),
                         "trained": paths.HEAD.exists(), "proof": proof(self.secret)})

    def do_POST(self):
        if not self._gate():
            return
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            return self.reply(415, {"error": "Content-Type must be application/json"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self.reply(400, {"error": "invalid Content-Length"})
        if not 0 < length <= MAX_BODY:
            return self.reply(413, {"error": f"body must be 1..{MAX_BODY} bytes"})
        try:
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
            started = time.perf_counter()
            out = handle(self.path, body)
            out["ms"] = round((time.perf_counter() - started) * 1000)
            self.reply(200, out)
        except LookupError:
            self.reply(404, {"error": "not found"})
        except (ValueError, TypeError, KeyError) as err:
            self.reply(400, {"error": str(err)})
        except Exception as err:  # report, don't die
            self.reply(500, {"error": f"{type(err).__name__}: {err}"})


def _watchdog(server):
    while True:
        time.sleep(15)
        if BRAIN.agent is not None and BRAIN.idle() > IDLE_UNLOAD:
            BRAIN.unload()
        if BRAIN.agent is None and BRAIN.idle() > EXIT_AFTER:
            server.shutdown()
            return


def main():
    Handler.secret = paths.token()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    threading.Thread(target=_watchdog, args=(server,), daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
