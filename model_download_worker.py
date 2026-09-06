from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from model_manager import download_asr, download_audio_preprocessor, download_silero

APP_DIR = Path(__file__).parent
DOWNLOAD_LOGGER_NAME = "LiveTranslate.Download"


def _configure_logging() -> tuple[logging.Logger, Path]:
    log_dir = APP_DIR / "logs" / "downloads"
    log_dir.mkdir(parents=True, exist_ok=True)
    detail_log = log_dir / f"download_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}.log"

    logger = logging.getLogger(DOWNLOAD_LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(logging.INFO)
    stream.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream)

    detail = logging.FileHandler(detail_log, encoding="utf-8")
    detail.setLevel(logging.DEBUG)
    detail.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(detail)

    # Third-party download libraries are intentionally not routed to stdout.
    # Their warnings/errors remain available in stderr or the detailed traceback
    # file if the worker fails, while the UI receives only our download status.
    logging.getLogger().handlers.clear()
    logging.getLogger().setLevel(logging.CRITICAL)
    for name in (
        "huggingface_hub",
        "modelscope",
        "urllib3",
        "httpx",
        "httpcore",
        "filelock",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    return logger, detail_log


def _download_one(item: dict, hub: str, proxy: str) -> None:
    model_type = str(item.get("type", ""))
    if model_type == "silero-vad":
        download_silero(proxy=proxy)
    elif model_type in (
        "sensevoice",
        "funasr-nano",
        "funasr-mlt-nano",
        "anime-whisper",
    ):
        download_asr(model_type, hub=hub, proxy=proxy)
    elif model_type.startswith("funasr:"):
        model_key = model_type.split(":", 1)[1]
        download_asr(
            "funasr",
            model_size=model_key,
            hub=hub,
            proxy=proxy,
        )
    elif model_type.startswith("whisper-"):
        size = model_type.removeprefix("whisper-")
        download_asr("whisper", model_size=size, hub=hub, proxy=proxy)
    elif model_type.startswith("preprocess:"):
        mode = model_type.split(":", 1)[1]
        download_audio_preprocessor(mode, proxy=proxy)
    else:
        raise ValueError(f"Unsupported download item: {model_type}")


def main() -> int:
    parser = argparse.ArgumentParser(description="LiveTranslate isolated model downloader")
    parser.add_argument("--spec", required=True, help="JSON encoded download specification")
    args = parser.parse_args()

    logger, detail_log = _configure_logging()
    try:
        spec = json.loads(args.spec)
        items = list(spec.get("items") or [])
        hub = str(spec.get("hub") or "ms")
        proxy = str(spec.get("proxy") or "system")
        if not items:
            raise ValueError("No download items were provided")

        logger.info(f"Download worker started (PID {os.getpid()})")
        logger.info(f"Detailed log: {detail_log}")
        for index, item in enumerate(items, 1):
            name = str(item.get("name") or item.get("type") or "model")
            logger.info(f"[{index}/{len(items)}] Preparing {name}")
            _download_one(item, hub, proxy)
            logger.info(f"[{index}/{len(items)}] Ready: {name}")
        logger.info("All requested models are ready")
        return 0
    except Exception as exc:
        logger.error(f"Download failed: {exc}")
        try:
            with detail_log.open("a", encoding="utf-8") as handle:
                handle.write("\n--- traceback ---\n")
                traceback.print_exc(file=handle)
        except OSError:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
