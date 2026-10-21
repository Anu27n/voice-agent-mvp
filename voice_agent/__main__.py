"""Run the server: ``python -m voice_agent`` (reads .env and the environment)."""

from __future__ import annotations

import logging
import os


def main() -> None:
    import uvicorn

    from .config import Settings
    from .main import create_app

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    settings = Settings.from_env()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
