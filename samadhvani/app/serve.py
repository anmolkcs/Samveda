"""Launcher for the demo server.

Kept separate from ``server.py`` so that importing the app (for tests, or for a
WSGI/ASGI host) never starts a server.

    python -m samadhvani.app.serve --port 8000
    samadhvani serve --port 8000

Binds to ``0.0.0.0`` by default: in a container or a notebook sandbox the preview
proxy reaches the process from outside, and a server bound to ``127.0.0.1`` is
invisible to it.
"""

from __future__ import annotations

import argparse
import os


def run(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> int:
    try:
        import uvicorn
    except ImportError:
        print("The demo needs FastAPI + uvicorn:\n  pip install 'samadhvani[app]'")
        return 1
    # `uvicorn.run` with an import string is required for --reload, and is the
    # friendlier form for a proxy either way.
    uvicorn.run(
        "samadhvani.app.server:app",
        host=host,
        port=int(port),
        reload=reload,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="samadhvani-serve",
                                 description="Run the Samadhvani chant demo.")
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args(argv)
    return run(args.host, args.port, args.reload)


if __name__ == "__main__":
    raise SystemExit(main())
