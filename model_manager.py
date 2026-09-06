import contextlib
import functools
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger("LiveTranslate.Download")

_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


@contextlib.contextmanager
def _proxy_env(proxy: str):
    """Temporarily route all download backends through a proxy.

    proxy:
        "system" / "" / None -> leave ambient env & OS proxy untouched
        "none"               -> force-disable any proxy for this download
        a URL                -> send urllib/requests/httpx traffic through it

    Covers torch.hub (urllib), huggingface_hub and modelscope (requests),
    which all honor the *_PROXY env vars; urllib additionally gets an explicit
    opener so a previously cached default opener cannot bypass the setting.
    """
    import urllib.request

    if proxy in ("system", "", None):
        yield
        return
    saved_env: dict = {key: os.environ.get(key) for key in _PROXY_ENV_KEYS}
    saved_no_proxy = os.environ.get("NO_PROXY")
    saved_opener = getattr(urllib.request, "_opener", None)
    try:
        if proxy == "none":
            for key in _PROXY_ENV_KEYS:
                os.environ.pop(key, None)
            os.environ["NO_PROXY"] = "*"
            handler = urllib.request.ProxyHandler({})
        else:
            for key in _PROXY_ENV_KEYS:
                os.environ[key] = proxy
            os.environ.pop("NO_PROXY", None)
            handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        urllib.request.install_opener(urllib.request.build_opener(handler))
        log.info(f"Download proxy active: {proxy}")
        yield
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if saved_no_proxy is None:
            os.environ.pop("NO_PROXY", None)
        else:
            os.environ["NO_PROXY"] = saved_no_proxy
        urllib.request.install_opener(saved_opener)

APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "models"
PREPROCESS_ENVS_DIR = APP_DIR / ".preprocess-envs"
PREPROCESS_MODELS_DIR = MODELS_DIR / "audio_preprocess"

# Optional machine-local overrides let multiple projects reuse the same heavy
# preprocessing runtimes without copying multi-gigabyte virtual environments.
# Values may point either to python.exe directly or to the environment root.
_PREPROCESS_ENV_PYTHON_KEYS = {
    "mdx_net": "LIVETRANSLATE_AUDIO_SEPARATOR_PYTHON",
    "melband_roformer": "LIVETRANSLATE_AUDIO_SEPARATOR_PYTHON",
}

_AUDIO_SEPARATOR_LIVE_MODES = ("mdx_net", "melband_roformer")

PREPROCESSOR_PROFILES = {
    "off": {
        "display_name": "Off",
        "estimated_bytes": 0,
    },
    "mdx_net": {
        "display_name": "MDX-NET (Low latency)",
        "model_filename": "UVR-MDX-NET-Inst_HQ_3.onnx",
        "min_model_bytes": 10_000_000,
        "estimated_bytes": 75_000_000,
        "runtime_estimated_bytes": 3_200_000_000,
        "shared_runtime_estimated_bytes": 650_000_000,
        "env": "audio-separator-live",
        "model_subdir": "audio-separator-live",
    },
    "melband_roformer": {
        "display_name": "MelBand RoFormer (High quality)",
        "model_filename": "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
        "min_model_bytes": 900_000_000,
        "estimated_bytes": 1_050_000_000,
        "runtime_estimated_bytes": 3_200_000_000,
        "shared_runtime_estimated_bytes": 650_000_000,
        "env": "audio-separator-live",
        "model_subdir": "audio-separator-live",
    },
}

ASR_MODEL_IDS = {
    "sensevoice": "iic/SenseVoiceSmall",
    "funasr-nano": "FunAudioLLM/Fun-ASR-Nano-2512",
    "funasr-mlt-nano": "FunAudioLLM/Fun-ASR-MLT-Nano-2512",
    "anime-whisper": "litagin/anime-whisper",
}

FUNASR_MODEL_PROFILES = {
    "sensevoice-small": {
        "display_name": "SenseVoice Small",
        "family": "sensevoice",
        "legacy_engine": "sensevoice",
        "modelscope_id": "iic/SenseVoiceSmall",
        "huggingface_id": "FunAudioLLM/SenseVoiceSmall",
        "estimated_bytes": 940_000_000,
        "supports_padding": True,
        "supports_language": True,
    },
    "funasr-nano-2512": {
        "display_name": "Fun-ASR-Nano",
        "family": "funasr-nano",
        "legacy_engine": "funasr-nano",
        "modelscope_id": "FunAudioLLM/Fun-ASR-Nano-2512",
        "huggingface_id": "FunAudioLLM/Fun-ASR-Nano-2512",
        # includes the separately-fetched Qwen3-0.6B weights (~1.5GB)
        "estimated_bytes": 3_500_000_000,
        "supports_padding": False,
        "supports_language": True,
    },
    "funasr-mlt-nano-2512": {
        "display_name": "Fun-ASR-MLT-Nano",
        "family": "funasr-nano",
        "legacy_engine": "funasr-mlt-nano",
        "modelscope_id": "FunAudioLLM/Fun-ASR-MLT-Nano-2512",
        "huggingface_id": "FunAudioLLM/Fun-ASR-MLT-Nano-2512",
        # includes the separately-fetched Qwen3-0.6B weights (~1.5GB)
        "estimated_bytes": 3_500_000_000,
        "supports_padding": False,
        "supports_language": True,
    },
}

DEFAULT_FUNASR_MODEL = "sensevoice-small"

FUNASR_LEGACY_ENGINE_ALIASES = {
    "sensevoice": "sensevoice-small",
    "funasr-nano": "funasr-nano-2512",
    "funasr-mlt-nano": "funasr-mlt-nano-2512",
}

# HuggingFace repo ids for engines whose namespace differs from ModelScope.
# SenseVoice lives under `iic/` on ModelScope but `FunAudioLLM/` on HuggingFace.
ASR_MODEL_IDS_HF = {
    "sensevoice": "FunAudioLLM/SenseVoiceSmall",
}


def asr_model_id(
    engine_type: str, hub: str = "ms", funasr_model: str | None = None
) -> str:
    """Return the repo id for an engine on the given hub ('ms' or 'hf')."""
    if engine_type == "funasr":
        return funasr_model_id(funasr_model, hub)
    if engine_type in FUNASR_LEGACY_ENGINE_ALIASES:
        return funasr_model_id(FUNASR_LEGACY_ENGINE_ALIASES[engine_type], hub)
    if hub == "hf" and engine_type in ASR_MODEL_IDS_HF:
        return ASR_MODEL_IDS_HF[engine_type]
    return ASR_MODEL_IDS[engine_type]

ASR_DISPLAY_NAMES = {
    "funasr": "FunASR",
    "sensevoice": "SenseVoice Small",
    "funasr-nano": "Fun-ASR-Nano",
    "funasr-mlt-nano": "Fun-ASR-MLT-Nano",
    "whisper": "Whisper",
    "anime-whisper": "Anime-Whisper",
    "remote-whisper": "Remote-Whisper",
}

_MODEL_SIZE_BYTES = {
    "silero-vad": 2_000_000,
    "sensevoice": 940_000_000,
    "funasr-nano": 1_050_000_000,
    "funasr-mlt-nano": 1_050_000_000,
    "whisper-tiny": 78_000_000,
    "whisper-base": 148_000_000,
    "whisper-small": 488_000_000,
    "whisper-medium": 1_530_000_000,
    "whisper-large-v3": 3_100_000_000,
    "anime-whisper": 3_100_000_000,
}

_WHISPER_SIZES = ["tiny", "base", "small", "medium", "large-v3"]

_CACHE_MODELS = [
    ("SenseVoice Small", "funasr", "sensevoice-small"),
    ("Fun-ASR-Nano", "funasr", "funasr-nano-2512"),
    ("Fun-ASR-MLT-Nano", "funasr", "funasr-mlt-nano-2512"),
    ("Anime-Whisper", "anime-whisper"),
]


def normalize_funasr_model_key(model_key: str | None) -> str:
    if model_key in FUNASR_MODEL_PROFILES:
        return model_key
    if model_key in FUNASR_LEGACY_ENGINE_ALIASES:
        return FUNASR_LEGACY_ENGINE_ALIASES[model_key]
    return DEFAULT_FUNASR_MODEL


def normalize_asr_engine_selection(
    engine_type: str | None, funasr_model: str | None = None
) -> tuple[str, str]:
    if engine_type in FUNASR_LEGACY_ENGINE_ALIASES:
        return "funasr", FUNASR_LEGACY_ENGINE_ALIASES[engine_type]
    if engine_type == "funasr":
        return "funasr", normalize_funasr_model_key(funasr_model)
    return engine_type or "funasr", normalize_funasr_model_key(funasr_model)


def migrate_funasr_settings(settings: dict | None) -> dict | None:
    if not settings:
        return settings
    engine, model_key = normalize_asr_engine_selection(
        settings.get("asr_engine"), settings.get("funasr_model")
    )
    settings["asr_engine"] = engine
    if engine == "funasr":
        settings["funasr_model"] = model_key
    else:
        settings.setdefault("funasr_model", DEFAULT_FUNASR_MODEL)
    return settings


def funasr_profile(model_key: str | None) -> dict:
    return FUNASR_MODEL_PROFILES[normalize_funasr_model_key(model_key)]


def funasr_model_options() -> list[tuple[str, str]]:
    return [
        (key, profile["display_name"])
        for key, profile in FUNASR_MODEL_PROFILES.items()
    ]


def funasr_display_name(model_key: str | None) -> str:
    return funasr_profile(model_key)["display_name"]


def funasr_supports_padding(model_key: str | None) -> bool:
    return bool(funasr_profile(model_key).get("supports_padding"))


def funasr_model_id(model_key: str | None, hub: str = "ms") -> str:
    profile = funasr_profile(model_key)
    return profile["huggingface_id"] if hub == "hf" else profile["modelscope_id"]


def _custom_whisper_path(value) -> Path | None:
    if not value or value in _WHISPER_SIZES:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = APP_DIR / path
    return path


def is_faster_whisper_model_dir(path) -> bool:
    """True when path looks like a CTranslate2 faster-whisper model directory."""
    if not path:
        return False
    path = Path(path)
    return (
        path.is_dir()
        and (path / "model.bin").is_file()
        and (path / "config.json").is_file()
    )


def resolve_custom_whisper_model(value) -> str | None:
    path = _custom_whisper_path(value)
    if path and is_faster_whisper_model_dir(path):
        return str(path.resolve())
    return None


def _is_builtin_whisper_cache(path: Path) -> bool:
    parts = set(path.parts)
    return any(f"models--Systran--faster-whisper-{s}" in parts for s in _WHISPER_SIZES)


def _hf_snapshot_name(path: Path) -> str | None:
    """Return 'org/repo' for .../models--org--repo/snapshots/<hash>."""
    if path.parent.name != "snapshots":
        return None
    repo_dir = path.parent.parent
    if not repo_dir.name.startswith("models--"):
        return None

    encoded = repo_dir.name.removeprefix("models--")
    parts = encoded.split("--", 1)
    if len(parts) != 2 or not all(parts):
        return None
    return f"{parts[0]}/{parts[1]}"


def list_local_faster_whisper_models() -> list[dict]:
    """Scan ./models for user-provided faster-whisper model directories."""
    if not MODELS_DIR.exists():
        return []

    entries = []
    name_counts = {}
    seen = set()
    try:
        model_bins = list(MODELS_DIR.rglob("model.bin"))
    except (OSError, PermissionError):
        return []

    for model_bin in model_bins:
        model_dir = model_bin.parent
        if _is_builtin_whisper_cache(model_dir):
            continue
        if not is_faster_whisper_model_dir(model_dir):
            continue
        try:
            resolved = str(model_dir.resolve())
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)

        name = _hf_snapshot_name(model_dir) or model_dir.name
        name_counts[name] = name_counts.get(name, 0) + 1
        if name_counts[name] > 1:
            name = f"{name} ({model_dir.name[:8]})"
        entries.append({"name": name, "path": resolved})

    entries.sort(key=lambda item: item["name"].lower())
    return entries

def local_faster_whisper_display_name(path) -> str | None:
    """Return the same display name used by the local Whisper model selector."""
    resolved = resolve_custom_whisper_model(path)
    if not resolved:
        return None
    for item in list_local_faster_whisper_models():
        if item["path"] == resolved:
            return item["name"]
    return _hf_snapshot_name(Path(resolved)) or Path(resolved).name

def apply_cache_env():
    """Point all model caches to ./models/."""
    resolved = str(MODELS_DIR.resolve())
    os.environ["MODELSCOPE_CACHE"] = os.path.join(resolved, "modelscope")
    os.environ["HF_HOME"] = os.path.join(resolved, "huggingface")
    os.environ["TORCH_HOME"] = os.path.join(resolved, "torch")
    log.info(f"Cache env set: {resolved}")


def _has_silero_pkg() -> bool:
    """True when the silero-vad PyPI package (model bundled in wheel) is installed."""
    import importlib.util

    return importlib.util.find_spec("silero_vad") is not None


def is_silero_cached() -> bool:
    if _has_silero_pkg():
        return True
    torch_hub = MODELS_DIR / "torch" / "hub"
    return any(torch_hub.glob("snakers4_silero-vad*")) if torch_hub.exists() else False


def _ms_model_path(org, name):
    """Return the first existing ModelScope cache path, or the default.

    Layouts by SDK version: {org}/{name} = <=1.37 with explicit cache_dir;
    models/{org}/{name} = 1.34~1.37 env-default cache, which >=1.38 keeps
    reusing as legacy even when cache_dir is passed (dots in names written
    as ___ by old SDKs); hub trees = older SDKs; models/{org}--{name}/
    snapshots/{revision} = >=1.38 fresh cache.
    """
    ms_root = MODELS_DIR / "modelscope"
    for sub in (
        ms_root / org / name,
        ms_root / "models" / org / name,
        ms_root / "models" / org / name.replace(".", "___"),
        ms_root / "hub" / "models" / org / name,
        ms_root / "hub" / org / name,
    ):
        if sub.exists():
            return sub
    snap_root = ms_root / "models" / f"{org}--{name}" / "snapshots"
    if snap_root.is_dir():
        snaps = sorted(d for d in snap_root.iterdir() if d.is_dir())
        if snaps:
            return snaps[-1]
    return ms_root / org / name


def _hf_repo_complete(org: str, name: str, min_bytes: int = 50_000_000) -> bool:
    """True if a HuggingFace repo cache exists AND finished downloading.

    A killed/aborted download leaves snapshot entries pointing at missing blobs
    (broken symlinks) or '.incomplete' blobs; treating that as cached makes the
    model load hang. Validate a snapshot where every file resolves (stat follows
    symlinks; a broken link raises) and the resolved bytes are substantial. This
    ignores orphan '.incomplete' blobs left behind by an earlier interrupted run.
    """
    snap_root = MODELS_DIR / "huggingface" / "hub" / f"models--{org}--{name}" / "snapshots"
    if not snap_root.exists():
        return False
    for snap in snap_root.iterdir():
        if not snap.is_dir():
            continue
        total = 0
        broken = False
        for f in snap.rglob("*"):
            if f.is_dir():
                continue
            try:
                total += f.stat().st_size
            except OSError:
                broken = True
                break
        if not broken and total >= min_bytes:
            return True
    return False


def is_asr_cached(engine_type, model_size="medium", hub="ms") -> bool:
    if engine_type == "funasr" or engine_type in FUNASR_LEGACY_ENGINE_ALIASES:
        model_key = (
            FUNASR_LEGACY_ENGINE_ALIASES[engine_type]
            if engine_type in FUNASR_LEGACY_ENGINE_ALIASES
            else normalize_funasr_model_key(model_size)
        )
        # Accept cache from either hub to avoid redundant downloads; the repo
        # namespace can differ between ModelScope and HuggingFace (SenseVoice).
        ms_org, ms_name = funasr_model_id(model_key, "ms").split("/")
        hf_org, hf_name = funasr_model_id(model_key, "hf").split("/")
        if not (
            _ms_model_path(ms_org, ms_name).exists()
            or _hf_repo_complete(hf_org, hf_name)
        ):
            return False
        # Nano's Qwen3-0.6B weights download separately; require them so the
        # download flow (not the deadline-bound worker) pulls them up-front.
        if funasr_profile(model_key)["family"] == "funasr-nano":
            model_dir = get_local_model_path(engine_type, hub, funasr_model=model_size)
            if not model_dir or not qwen_weights_present(model_dir):
                return False
        return True
    if engine_type == "anime-whisper":
        # HF-only (not published to ModelScope). Check that snapshots dir actually
        # contains weight files; an .incomplete blob means a prior run aborted mid-download.
        model_id = ASR_MODEL_IDS[engine_type]
        org, name = model_id.split("/")
        snap_root = (
            MODELS_DIR / "huggingface" / "hub" / f"models--{org}--{name}" / "snapshots"
        )
        if not snap_root.exists():
            return False
        for snap in snap_root.iterdir():
            if not snap.is_dir():
                continue
            has_weights = any(
                (snap / fn).exists()
                for fn in ("model.safetensors", "pytorch_model.bin")
            )
            has_config = (snap / "config.json").exists()
            if has_weights and has_config:
                return True
        return False
    elif engine_type == "whisper":
        if model_size not in _WHISPER_SIZES:
            return resolve_custom_whisper_model(model_size) is not None
        min_bytes = int(
            _MODEL_SIZE_BYTES.get(f"whisper-{model_size}", 50_000_000) * 0.5
        )
        return _hf_repo_complete(
            "Systran", f"faster-whisper-{model_size}", min_bytes=min_bytes
        )
    return True


def normalize_audio_preprocess_mode(mode: str | None) -> str:
    mode = str(mode or "off")
    return mode if mode in PREPROCESSOR_PROFILES else "off"


def audio_preprocessor_display_name(mode: str | None) -> str:
    mode = normalize_audio_preprocess_mode(mode)
    return PREPROCESSOR_PROFILES[mode]["display_name"]


def audio_preprocessor_model_dir(mode: str | None) -> Path:
    mode = normalize_audio_preprocess_mode(mode)
    profile = PREPROCESSOR_PROFILES[mode]
    return PREPROCESS_MODELS_DIR / str(profile.get("model_subdir", mode))


def audio_preprocessor_ready_marker(mode: str | None) -> Path:
    mode = normalize_audio_preprocess_mode(mode)
    model_dir = audio_preprocessor_model_dir(mode)
    if mode in _AUDIO_SEPARATOR_LIVE_MODES:
        return model_dir / f".ready-{mode}"
    return model_dir / ".ready"


def _external_audio_preprocessor_python(mode: str | None) -> Path | None:
    mode = normalize_audio_preprocess_mode(mode)
    env_key = _PREPROCESS_ENV_PYTHON_KEYS.get(mode)
    if not env_key:
        return None
    value = os.environ.get(env_key)
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_dir():
        path = path / "Scripts" / "python.exe"
    return path


def _major_minor(version: str) -> tuple[int, int] | None:
    try:
        parts = version.split("+", 1)[0].split(".")
        return int(parts[0]), int(parts[1])
    except (IndexError, TypeError, ValueError):
        return None


@functools.lru_cache(maxsize=1)
def _shared_torch_runtime() -> dict | None:
    """Return the app's main Torch runtime when an overlay can safely reuse it."""
    python = APP_DIR / ".venv" / "Scripts" / "python.exe"
    if not python.is_file():
        return None

    # Metadata probing is intentionally cheap: importing CUDA Torch while the
    # live app is using the GPU can take tens of seconds and false-negative the
    # shared-runtime check. The overlay performs a real import validation after
    # installing its small package set.
    probe = (
        "import importlib.metadata as m, json, sysconfig; "
        "print(json.dumps({"
        "'torch': m.version('torch'), "
        "'torchaudio': m.version('torchaudio'), "
        "'site_packages': sysconfig.get_paths()['purelib']"
        "}))"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", probe],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        info = json.loads(result.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None

    torch_mm = _major_minor(str(info.get("torch", "")))
    torchaudio_mm = _major_minor(str(info.get("torchaudio", "")))
    if torch_mm is None or torch_mm < (2, 1) or torchaudio_mm != torch_mm:
        return None
    site_packages = Path(str(info.get("site_packages", "")))
    if not site_packages.is_dir():
        return None

    version = str(info.get("torch", ""))
    info["cuda"] = version.split("+", 1)[1] if "+" in version else "cpu"
    info["python"] = python
    info["site_packages"] = site_packages
    return info


def audio_preprocessor_env_python(mode: str | None) -> Path | None:
    mode = normalize_audio_preprocess_mode(mode)
    external = _external_audio_preprocessor_python(mode)
    if external is not None:
        return external
    env_name = PREPROCESSOR_PROFILES[mode].get("env")
    if not env_name:
        return None
    return PREPROCESS_ENVS_DIR / env_name / "Scripts" / "python.exe"



def _audio_preprocessor_model_present(mode: str | None) -> bool:
    mode = normalize_audio_preprocess_mode(mode)
    if mode == "off":
        return True
    model_dir = audio_preprocessor_model_dir(mode)
    if mode == "mdx_net":
        profile = PREPROCESSOR_PROFILES[mode]
        model = model_dir / profile["model_filename"]
        mdx_data = model_dir / "mdx_model_data.json"
        vr_data = model_dir / "vr_model_data.json"
        return (
            model.is_file()
            and model.stat().st_size > int(profile["min_model_bytes"])
            and mdx_data.is_file()
            and mdx_data.stat().st_size > 0
            and vr_data.is_file()
            and vr_data.stat().st_size > 0
        )
    if mode == "melband_roformer":
        profile = PREPROCESSOR_PROFILES[mode]
        model = model_dir / profile["model_filename"]
        yaml_file = model.with_suffix(".yaml")
        return (
            model.is_file()
            and model.stat().st_size > int(profile["min_model_bytes"])
            and yaml_file.is_file()
            and yaml_file.stat().st_size > 0
        )
    return False


def is_audio_preprocessor_ready(mode: str | None) -> bool:
    mode = normalize_audio_preprocess_mode(mode)
    if mode == "off":
        return True

    marker = audio_preprocessor_ready_marker(mode)
    if not marker.exists():
        return False
    if not _audio_preprocessor_model_present(mode):
        return False

    env_python = audio_preprocessor_env_python(mode)
    if env_python is None or not env_python.is_file():
        return False
    return mode in _AUDIO_SEPARATOR_LIVE_MODES


def get_missing_audio_preprocessor(mode: str | None) -> list[dict]:
    mode = normalize_audio_preprocess_mode(mode)
    if mode == "off" or is_audio_preprocessor_ready(mode):
        return []
    profile = PREPROCESSOR_PROFILES[mode]
    estimated = 0
    if not _audio_preprocessor_model_present(mode):
        estimated += int(profile.get("estimated_bytes", 0))
    external_python = _external_audio_preprocessor_python(mode)
    env_name = profile.get("env")
    runtime_missing = False
    if env_name:
        if external_python is not None:
            runtime_missing = not external_python.exists()
        else:
            runtime_missing = not (PREPROCESS_ENVS_DIR / env_name / ".deps-ready").exists()
    if runtime_missing:
        runtime_estimated = int(profile.get("runtime_estimated_bytes", 0))
        if mode in _AUDIO_SEPARATOR_LIVE_MODES and _shared_torch_runtime() is not None:
            runtime_estimated = int(
                profile.get("shared_runtime_estimated_bytes", runtime_estimated)
            )
        estimated += runtime_estimated
    return [
        {
            "name": profile["display_name"],
            "type": f"preprocess:{mode}",
            "estimated_bytes": estimated,
        }
    ]


def _uv_executable() -> str:
    bundled = APP_DIR / "tools" / "uv.exe"
    if bundled.is_file():
        return str(bundled)
    local = APP_DIR / ".venv" / "Scripts" / "uv.exe"
    if local.is_file():
        return str(local)
    found = shutil.which("uv")
    if found:
        return found
    raise RuntimeError("uv is required for optional audio preprocessing runtimes")


def _run_logged(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    progress_label: str | None = None,
) -> None:
    log.info("Running: " + " ".join(map(str, cmd)))
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", str(APP_DIR / ".uv-cache"))
    env.setdefault("UV_PYTHON_INSTALL_DIR", str(APP_DIR / "tools" / "python"))
    env.setdefault("UV_LINK_MODE", "copy")
    env.setdefault("UV_NO_PROGRESS", "1")
    env.setdefault("TMP", str(APP_DIR / ".tmp"))
    env.setdefault("TEMP", str(APP_DIR / ".tmp"))
    Path(env["UV_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["TMP"]).mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None

    lines: queue.Queue[str | None] = queue.Queue()

    def _read_output() -> None:
        try:
            for line in proc.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    threading.Thread(target=_read_output, daemon=True).start()
    started = time.monotonic()
    last_heartbeat = started
    output_done = False
    while not output_done:
        try:
            line = lines.get(timeout=1.0)
        except queue.Empty:
            line = ""
        if line is None:
            output_done = True
        elif line:
            text = line.rstrip()
            if text:
                log.info(text)
        now = time.monotonic()
        if (
            progress_label
            and proc.poll() is None
            and now - last_heartbeat >= 15.0
        ):
            elapsed = int(now - started)
            log.info(f"{progress_label}: still working ({elapsed}s elapsed)")
            last_heartbeat = now
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {cmd[0]}")


def _download_url(url: str, target: Path, label: str) -> None:
    import urllib.request
    import uuid

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(f"{target.name}.{uuid.uuid4().hex}.part")
    last_bucket = [-1]

    def _progress(downloaded: int, total_size: int):
        if total_size <= 0:
            return
        downloaded = min(downloaded, total_size)
        pct = int(downloaded * 100 / total_size)
        bucket = min(10, pct // 10)
        if bucket != last_bucket[0]:
            last_bucket[0] = bucket
            log.info(f"{label}: {min(bucket * 10, 100)}%")

    log.info(f"Downloading {label}...")
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "LiveTranslateWithNC/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response, part.open(
            "wb"
        ) as handle:
            try:
                total_size = int(response.headers.get("Content-Length") or 0)
            except (TypeError, ValueError):
                total_size = 0
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                _progress(downloaded, total_size)
        if part.stat().st_size == 0:
            raise RuntimeError(f"{label} download returned an empty file")
        part.replace(target)
    finally:
        try:
            part.unlink(missing_ok=True)
        except OSError:
            # Windows AV/indexers can briefly hold a just-written temp file.
            # A stale uniquely-named .part must not break later downloads.
            pass



def _torch_index_for_host() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            cc = float(result.stdout.splitlines()[0].strip())
            return "cu128" if cc >= 12.0 else "cu126"
    except Exception:
        pass
    return "cpu"


def _resolve_shared_overlay_packages(
    uv: str,
    main_python: Path,
    requirements: list[str],
    *,
    label: str,
) -> list[str]:
    """Resolve only packages that the main environment does not already satisfy."""
    cmd = [
        uv,
        "pip",
        "install",
        "--system-certs",
        "--dry-run",
        "--color",
        "never",
        "--python",
        str(main_python),
        *requirements,
    ]
    log.info(f"Comparing {label} dependencies with the main LiveTranslate environment...")
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", str(APP_DIR / ".uv-cache"))
    env.setdefault("UV_PYTHON_INSTALL_DIR", str(APP_DIR / "tools" / "python"))
    env.setdefault("UV_NO_PROGRESS", "1")
    result = subprocess.run(
        [str(x) for x in cmd],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )
    for line in result.stdout.splitlines():
        text = line.strip()
        if text:
            log.info(text)
    if result.returncode != 0:
        raise RuntimeError(f"Could not resolve shared dependencies for {label}")

    package_re = re.compile(r"^\+\s+([A-Za-z0-9_.-]+)==([^\s]+)$")
    planned: list[str] = []
    for line in result.stdout.splitlines():
        match = package_re.match(line.strip())
        if match:
            planned.append(f"{match.group(1)}=={match.group(2)}")
    log.info(
        f"{label}: main environment can be reused; "
        f"isolated overlay needs {len(planned)} package(s)"
    )
    return planned


def _install_shared_preprocessor_overlay(
    mode: str,
    uv: str,
    env_dir: Path,
    python: Path,
    shared_torch: dict,
) -> None:
    """Keep an isolated venv while reusing compatible packages from the main venv."""
    profile = PREPROCESSOR_PROFILES[mode]
    site_packages = env_dir / "Lib" / "site-packages"
    site_packages.mkdir(parents=True, exist_ok=True)
    main_site_packages = Path(shared_torch["site_packages"])
    relative_main_site = os.path.relpath(main_site_packages, site_packages)
    (site_packages / "livetranslate-main-runtime.pth").write_text(
        relative_main_site + "\n", encoding="utf-8"
    )

    cuda = shared_torch.get("cuda") or "CPU"
    log.info(
        f"Using isolated {profile['display_name']} overlay with shared main runtime: "
        f"torch={shared_torch['torch']}, "
        f"torchaudio={shared_torch['torchaudio']}, CUDA={cuda}"
    )

    if mode in _AUDIO_SEPARATOR_LIVE_MODES:
        # ORT 1.27+ PyPI wheels switched to CUDA 13.0. LiveTranslate's current
        # NVIDIA runtime is CUDA 12.8, so keep the GPU EP on the official
        # CUDA-12.8-compatible ORT release line (1.21.x-1.26.x).
        requirements = [
            "audio-separator[gpu]==0.47.0",
            "onnxruntime-gpu>=1.21,<1.27",
            # audio-separator imports audioread directly, but its 0.47.0
            # metadata does not guarantee that package is present when librosa
            # is reused from LiveTranslate's main environment.
            "audioread>=3,<4",
        ]
        validate = (
            "import torch, numpy, scipy, onnxruntime; "
            "assert tuple(map(int, onnxruntime.__version__.split('.')[:2])) < (1,27), "
            "'onnxruntime-gpu 1.27+ requires CUDA 13'; "
            "from audio_separator.separator import Separator; "
            "print('audio-separator live runtime ready: torch=' + torch.__version__ + "
            "', onnxruntime=' + onnxruntime.__version__ + "
            "', providers=' + ','.join(onnxruntime.get_available_providers()))"
        )
    else:
        raise ValueError(f"Unsupported shared overlay mode: {mode}")

    planned = _resolve_shared_overlay_packages(
        uv,
        Path(shared_torch["python"]),
        requirements,
        label=profile["display_name"],
    )
    if planned:
        log.info(
            f"Installing {profile['display_name']} isolated overlay packages: "
            + ", ".join(planned)
        )
        _run_logged(
            [
                uv,
                "pip",
                "install",
                "--system-certs",
                "--no-deps",
                "--python",
                str(python),
                *planned,
            ],
            progress_label=f"Installing {profile['display_name']} overlay",
        )
    _run_logged(
        [str(python), "-c", validate],
        progress_label=f"Validating shared {profile['display_name']} runtime",
    )


def _ensure_audio_preprocessor_env(mode: str) -> Path:
    mode = normalize_audio_preprocess_mode(mode)
    profile = PREPROCESSOR_PROFILES[mode]
    env_name = profile.get("env")
    if not env_name:
        raise ValueError(f"{mode} does not use a Python runtime")

    external = _external_audio_preprocessor_python(mode)
    if external is not None:
        if not external.is_file():
            env_key = _PREPROCESS_ENV_PYTHON_KEYS[mode]
            raise FileNotFoundError(
                f"Configured external preprocessing runtime does not exist: "
                f"{env_key}={external}"
            )
        log.info(f"Using shared preprocessing runtime: {external}")
        return external

    uv = _uv_executable()
    env_dir = PREPROCESS_ENVS_DIR / env_name
    python = env_dir / "Scripts" / "python.exe"
    deps_marker = env_dir / ".deps-ready"
    PREPROCESS_ENVS_DIR.mkdir(parents=True, exist_ok=True)
    if not python.exists():
        log.info(f"Creating uv environment for {profile['display_name']}...")
        _run_logged(
            [uv, "venv", "--python", "3.12", str(env_dir)],
            progress_label=f"Creating {profile['display_name']} runtime",
        )
    if deps_marker.exists():
        return python

    if mode in _AUDIO_SEPARATOR_LIVE_MODES:
        shared_torch = _shared_torch_runtime()
        if shared_torch is not None:
            _install_shared_preprocessor_overlay(mode, uv, env_dir, python, shared_torch)
            deps_marker.write_text("shared-main-runtime\n", encoding="ascii")
            return python

    torch_index = _torch_index_for_host()
    index_url = (
        "https://download.pytorch.org/whl/cpu"
        if torch_index == "cpu"
        else f"https://download.pytorch.org/whl/{torch_index}"
    )
    log.info(f"Installing PyTorch runtime ({torch_index})...")
    _run_logged(
        [
            uv,
            "pip",
            "install",
            "--system-certs",
            "--python",
            str(python),
            "torch==2.8.0",
            "torchaudio==2.8.0",
            "--index-url",
            index_url,
        ],
        progress_label=f"Installing PyTorch runtime ({torch_index})",
    )
    if mode in _AUDIO_SEPARATOR_LIVE_MODES:
        packages = [
            "audio-separator[gpu]==0.47.0",
            "onnxruntime-gpu>=1.21,<1.27",
            "audioread>=3,<4",
        ]
    else:
        packages = []
    if packages:
        log.info(f"Installing {profile['display_name']} dependencies with uv...")
        _run_logged(
            [uv, "pip", "install", "--system-certs", "--python", str(python), *packages],
            progress_label=f"Installing {profile['display_name']} dependencies",
        )
    deps_marker.write_text("ready\n", encoding="ascii")
    return python


def download_audio_preprocessor(mode: str | None, proxy: str = "system") -> None:
    mode = normalize_audio_preprocess_mode(mode)
    if mode == "off":
        return
    profile = PREPROCESSOR_PROFILES[mode]
    model_dir = audio_preprocessor_model_dir(mode)
    model_dir.mkdir(parents=True, exist_ok=True)
    marker = audio_preprocessor_ready_marker(mode)
    reusable_model = marker.exists() and _audio_preprocessor_model_present(mode)
    marker.unlink(missing_ok=True)

    with _proxy_env(proxy):
        if mode not in _AUDIO_SEPARATOR_LIVE_MODES:
            raise ValueError(f"Unsupported audio preprocessing mode: {mode}")
        python = _ensure_audio_preprocessor_env(mode)
        if not reusable_model:
            model_filename = str(profile["model_filename"])
            # audio-separator downloads directly to the final filename. If the
            # process was interrupted, an incomplete .onnx/.ckpt can therefore
            # remain. A model without our ready marker is never trusted/reused.
            target = model_dir / model_filename
            target.unlink(missing_ok=True)
            if mode == "melband_roformer":
                target.with_suffix(".yaml").unlink(missing_ok=True)
            log.info(f"Downloading {profile['display_name']} model: {model_filename}")
            code = (
                "import logging,sys; "
                "from pathlib import Path; "
                "from audio_separator.separator import Separator; "
                "root=Path(sys.argv[1]).resolve(); root.mkdir(parents=True, exist_ok=True); "
                "sep=Separator(log_level=logging.INFO, model_file_dir=str(root), info_only=True); "
                "sep.download_model_and_data(sys.argv[2]); "
                "print('audio-separator model ready: ' + sys.argv[2])"
            )
            _run_logged(
                [str(python), "-c", code, str(model_dir), model_filename],
                cwd=APP_DIR,
                progress_label=f"Downloading {profile['display_name']} model",
            )
        else:
            log.info(f"{profile['display_name']} model already cached; reusing it")

    marker.write_text("ready\n", encoding="ascii")
    if not is_audio_preprocessor_ready(mode):
        marker.unlink(missing_ok=True)
        raise RuntimeError(f"{profile['display_name']} validation failed after download")
    # Match the portable installer policy: keep installed runtimes/models, not a
    # second copy of large CUDA wheels in uv's download cache.
    shutil.rmtree(APP_DIR / ".uv-cache", ignore_errors=True)
    shutil.rmtree(APP_DIR / ".tmp", ignore_errors=True)
    log.info(f"Audio preprocessor ready: {profile['display_name']}")


def get_missing_models(engine, model_size, hub) -> list:
    missing = []
    if not is_silero_cached():
        missing.append(
            {
                "name": "Silero VAD",
                "type": "silero-vad",
                "estimated_bytes": _MODEL_SIZE_BYTES["silero-vad"],
            }
        )
    if not is_asr_cached(engine, model_size, hub):
        if engine == "whisper" and model_size not in _WHISPER_SIZES:
            return missing
        if engine == "funasr" or engine in FUNASR_LEGACY_ENGINE_ALIASES:
            model_key = (
                FUNASR_LEGACY_ENGINE_ALIASES[engine]
                if engine in FUNASR_LEGACY_ENGINE_ALIASES
                else normalize_funasr_model_key(model_size)
            )
            profile = funasr_profile(model_key)
            key = f"funasr:{model_key}"
            display = profile["display_name"]
            estimated_bytes = profile["estimated_bytes"]
        elif engine == "whisper":
            key = engine if engine != "whisper" else f"whisper-{model_size}"
            display = f"Whisper {model_size}"
            estimated_bytes = _MODEL_SIZE_BYTES.get(key, 0)
        else:
            key = engine
            display = ASR_DISPLAY_NAMES.get(engine, engine)
            estimated_bytes = _MODEL_SIZE_BYTES.get(key, 0)
        missing.append(
            {
                "name": display,
                "type": key,
                "estimated_bytes": estimated_bytes,
            }
        )
    return missing


def get_local_model_path(engine_type, hub="ms", funasr_model: str | None = None):
    """Return local snapshot path if model is cached, else None.

    Checks the preferred hub first, then falls back to the other hub.
    """
    if engine_type == "funasr" or engine_type in FUNASR_LEGACY_ENGINE_ALIASES:
        model_key = (
            FUNASR_LEGACY_ENGINE_ALIASES[engine_type]
            if engine_type in FUNASR_LEGACY_ENGINE_ALIASES
            else normalize_funasr_model_key(funasr_model)
        )
        ms_org, ms_name = funasr_model_id(model_key, "ms").split("/")
        hf_org, hf_name = funasr_model_id(model_key, "hf").split("/")
    elif engine_type in ASR_MODEL_IDS:
        ms_org, ms_name = asr_model_id(engine_type, "ms").split("/")
        hf_org, hf_name = asr_model_id(engine_type, "hf").split("/")
    else:
        return None

    def _try_ms():
        local = _ms_model_path(ms_org, ms_name)
        return str(local) if local.exists() else None

    def _try_hf():
        snap_dir = (
            MODELS_DIR
            / "huggingface"
            / "hub"
            / f"models--{hf_org}--{hf_name}"
            / "snapshots"
        )
        if snap_dir.exists():
            snaps = sorted(snap_dir.iterdir())
            if snaps:
                return str(snaps[-1])
        return None

    if hub == "ms":
        return _try_ms() or _try_hf()
    else:
        return _try_hf() or _try_ms()


def download_silero(proxy: str = "system"):
    if _has_silero_pkg():
        log.info("Silero VAD bundled by silero-vad package, no download needed")
        return
    import torch

    log.info("Downloading Silero VAD...")
    with _proxy_env(proxy):
        try:
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad:master",
                model="silero_vad",
                trust_repo=True,
            )
        except Exception as exc:
            if "CERTIFICATE_VERIFY" not in str(exc):
                raise
            log.warning("SSL strict verification failed, retrying with relaxed flags")
            model, _ = _load_silero_relaxed_ssl()
    del model
    log.info("Silero VAD downloaded")


def _load_silero_relaxed_ssl():
    # Python 3.13 enables VERIFY_X509_STRICT by default, rejecting certificates
    # without an Authority Key Identifier (common behind SSL-inspecting proxies).
    import ssl

    import torch

    strict = getattr(ssl, "VERIFY_X509_STRICT", 0)
    original = ssl._create_default_https_context

    def relaxed_context(*args, **kwargs):
        ctx = ssl.create_default_context(*args, **kwargs)
        ctx.verify_flags &= ~strict
        return ctx

    ssl._create_default_https_context = relaxed_context
    try:
        return torch.hub.load(
            repo_or_dir="snakers4/silero-vad:master",
            model="silero_vad",
            trust_repo=True,
            force_reload=True,
        )
    finally:
        ssl._create_default_https_context = original


def qwen_weights_present(model_dir) -> bool:
    """Whether a nano model's embedded Qwen3-0.6B weights are in place.

    Nano repos ship the Qwen3-0.6B config but not its weights. A variant without
    the subdir needs no Qwen weights, so absence of the subdir counts as present.
    """
    qwen_dir = Path(model_dir) / "Qwen3-0.6B"
    if not qwen_dir.is_dir():
        return True
    return any(f.suffix in (".safetensors", ".bin") for f in qwen_dir.iterdir())


def ensure_qwen_weights(model_dir, hub: str = "ms") -> None:
    """Fetch Qwen3-0.6B weights into a nano model's embedded subdir (one-time).

    Kept off the ASR worker startup path: its 180s ready timeout would otherwise
    kill the process mid-download on slow links.
    """
    qwen_dir = Path(model_dir) / "Qwen3-0.6B"
    if not qwen_dir.is_dir():
        return
    if any(f.suffix in (".safetensors", ".bin") for f in qwen_dir.iterdir()):
        return
    log.info("Downloading Qwen3-0.6B weights (one-time)...")
    if hub == "hf":
        from huggingface_hub import snapshot_download
    else:
        from modelscope import snapshot_download

    snapshot_download(
        "Qwen/Qwen3-0.6B",
        local_dir=str(qwen_dir),
        ignore_patterns=["*.gguf"],
    )
    log.info("Qwen3-0.6B weights downloaded")


def download_asr(engine, model_size="medium", hub="ms", proxy="system"):
    resolved = str(MODELS_DIR.resolve())
    ms_cache = os.path.join(resolved, "modelscope")
    hf_cache = os.path.join(resolved, "huggingface", "hub")
    with _proxy_env(proxy):
        if engine == "funasr" or engine in FUNASR_LEGACY_ENGINE_ALIASES:
            model_key = (
                FUNASR_LEGACY_ENGINE_ALIASES[engine]
                if engine in FUNASR_LEGACY_ENGINE_ALIASES
                else normalize_funasr_model_key(model_size)
            )
            if hub == "ms":
                from modelscope import snapshot_download

                model_id = funasr_model_id(model_key, "ms")
                log.info(f"Downloading {model_id} from ModelScope...")
                snapshot_download(model_id=model_id, cache_dir=ms_cache)
            else:
                from huggingface_hub import snapshot_download

                model_id = funasr_model_id(model_key, "hf")
                log.info(f"Downloading {model_id} from HuggingFace...")
                snapshot_download(repo_id=model_id, cache_dir=hf_cache)
            funasr_dir = get_local_model_path("funasr", hub=hub, funasr_model=model_key)
            neutralize_funasr_requirements(funasr_dir)
            if funasr_dir and funasr_profile(model_key)["family"] == "funasr-nano":
                ensure_qwen_weights(funasr_dir, hub=hub)
        elif engine == "anime-whisper":
            # HF-only, ignore hub setting
            from huggingface_hub import snapshot_download

            model_id = ASR_MODEL_IDS[engine]
            log.info(f"Downloading {model_id} from HuggingFace...")
            snapshot_download(repo_id=model_id, cache_dir=hf_cache)
        elif engine == "whisper":
            if model_size not in _WHISPER_SIZES:
                raise ValueError(f"Invalid local faster-whisper model: {model_size}")
            from huggingface_hub import snapshot_download

            model_id = f"Systran/faster-whisper-{model_size}"
            log.info(f"Downloading {model_id} from HuggingFace...")
            snapshot_download(repo_id=model_id, cache_dir=hf_cache)
    log.info(f"ASR model downloaded: {engine}")


def neutralize_funasr_requirements(model_dir) -> None:
    """Skip FunASR's load-time `pip install -r requirements.txt`.

    With trust_remote_code=True, FunASR detects requirements.txt in the model
    dir and runs pip in a subprocess whose output is swallowed (PIPE). On a slow
    or proxy-blocked PyPI this hangs indefinitely with no log output, and it can
    pull heavy unused deps (e.g. gradio). All real deps already live in the venv,
    so rename the file out of the way to make the check miss.
    """
    if not model_dir:
        return
    req = Path(model_dir) / "requirements.txt"
    if req.exists():
        try:
            req.replace(req.with_name("requirements.txt.bundled"))
            log.info(f"Skipped FunASR requirements install: {req}")
        except OSError as exc:
            log.warning(f"Failed to neutralize {req}: {exc}")


def dir_size(path) -> int:
    total = 0
    try:
        for f in Path(path).rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except (OSError, PermissionError):
        pass
    return total


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.1f} MB"
    else:
        return f"{size_bytes / (1024**3):.2f} GB"


def get_cache_entries():
    """Scan ./models/ for cached models."""
    entries = []
    hf_base = MODELS_DIR / "huggingface" / "hub"
    torch_base = MODELS_DIR / "torch" / "hub"

    for entry in _CACHE_MODELS:
        if len(entry) == 3:
            name, engine, model_key = entry
        else:
            name, engine = entry
            model_key = None
        ms_org, ms_model = asr_model_id(engine, "ms", model_key).split("/")
        hf_org, hf_model = asr_model_id(engine, "hf", model_key).split("/")
        ms_path = _ms_model_path(ms_org, ms_model)
        hf_path = hf_base / f"models--{hf_org}--{hf_model}"
        if ms_path.exists():
            entries.append((f"{name} (ModelScope)", ms_path))
        if hf_path.exists():
            entries.append((f"{name} (HuggingFace)", hf_path))

    for size in _WHISPER_SIZES:
        hf_path = hf_base / f"models--Systran--faster-whisper-{size}"
        if hf_path.exists() and is_asr_cached("whisper", size, "hf"):
            entries.append((f"Whisper {size}", hf_path))

    for item in list_local_faster_whisper_models():
        entries.append((f"Whisper Local: {item['name']}", Path(item["path"])))

    if torch_base.exists():
        for d in sorted(torch_base.glob("snakers4_silero-vad*")):
            if d.is_dir():
                entries.append(("Silero VAD", d))
                break

    for mode, profile in PREPROCESSOR_PROFILES.items():
        if mode == "off":
            continue
        model_dir = audio_preprocessor_model_dir(mode)
        if model_dir.exists():
            entries.append((f"{profile['display_name']} (Audio Preprocess)", model_dir))
        env_name = profile.get("env")
        if env_name:
            env_dir = PREPROCESS_ENVS_DIR / env_name
            if env_dir.exists():
                entries.append((f"{profile['display_name']} Runtime", env_dir))

    return entries
