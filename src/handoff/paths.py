"""Where handoff keeps your data. Everything lives in one private folder on this machine."""
import os
import secrets
from pathlib import Path

HOME = Path(os.environ.get("HANDOFF_HOME", Path.home() / ".handoff")).expanduser()
PROMPTS = HOME / "data" / "prompts.jsonl"
LABELS = HOME / "data" / "labels.jsonl"
HEAD = HOME / "model" / "router.pt"
REPORT = HOME / "model" / "report.json"
LOG = HOME / "logs" / "server.log"
TOKEN = HOME / "token"
PIDFILE = HOME / "service.pid"
MARKER = HOME / ".handoff-home"


def _check_home() -> None:
    """Refuse to take over a folder that isn't ours (we chmod it to 0700)."""
    resolved = HOME.resolve()
    if resolved in (Path.home().resolve(), Path("/"), Path("/tmp").resolve()):
        raise SystemExit(f"HANDOFF_HOME={HOME} is not a dedicated folder — pick a new one like ~/.handoff")
    if HOME.exists() and not MARKER.exists() and any(HOME.iterdir()):
        raise SystemExit(f"{HOME} already has other files in it — point HANDOFF_HOME at a new, empty folder")


def ensure(path: Path) -> Path:
    """Create the parent folder (and HOME), private to this user (0700)."""
    _check_home()
    path.parent.mkdir(parents=True, exist_ok=True)
    MARKER.touch(exist_ok=True)
    for folder in {HOME, path.parent}:
        os.chmod(folder, 0o700)
    return path


def private(path: Path) -> Path:
    """Make a file readable by this user only (0600)."""
    if path.exists():
        os.chmod(path, 0o600)
    return path


def token() -> str:
    """Shared secret between the CLI and the local service; created on first use."""
    if not TOKEN.exists():
        ensure(TOKEN).write_text(secrets.token_hex(24))
        private(TOKEN)
    return TOKEN.read_text().strip()
