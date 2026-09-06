from pathlib import Path
import threading

import numpy as np
import pytest
import yaml

import model_manager
from audio_preprocessor import AudioPreprocessor


def test_audio_preprocessing_defaults_to_off():
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))

    assert config["audio"]["preprocess_mode"] == "off"
    assert model_manager.normalize_audio_preprocess_mode(None) == "off"
    assert model_manager.normalize_audio_preprocess_mode("unknown") == "off"


def test_supported_audio_preprocessing_modes_are_registered():
    assert set(model_manager.PREPROCESSOR_PROFILES) == {
        "off",
        "demucs_v4",
        "clearvoice_mossformer2_se",
    }


def test_off_mode_preserves_capture_pcm_contract():
    chunk = np.linspace(-0.5, 0.5, 512, dtype=np.float32)
    preprocessor = AudioPreprocessor("off", sample_rate=16000, chunk_duration=0.032)

    output = preprocessor.process_chunk(chunk)

    assert len(output) == 1
    np.testing.assert_array_equal(output[0], chunk)
    assert output[0].dtype == np.float32


@pytest.mark.parametrize(
    "mode", ["demucs_v4", "clearvoice_mossformer2_se"]
)
def test_buffered_preprocessor_returns_original_32ms_chunk_shape(monkeypatch, mode):
    """Exercise buffering/async scheduling without loading an external model."""
    preprocessor = AudioPreprocessor(mode, sample_rate=16000, chunk_duration=0.032)
    monkeypatch.setattr(preprocessor, "start", lambda: None)
    monkeypatch.setattr(
        preprocessor,
        "_request",
        lambda samples: np.asarray(samples, dtype=np.float32) * np.float32(0.5),
    )
    preprocessor._processing_thread = threading.Thread(
        target=preprocessor._processing_loop, daemon=True
    )
    preprocessor._processing_thread.start()

    source_chunks = [
        np.full(512, i / 100.0, dtype=np.float32)
        for i in range(preprocessor.window_chunks)
    ]
    output_chunks = []
    for chunk in source_chunks:
        output_chunks.extend(preprocessor.process_chunk(chunk))
    output_chunks.extend(preprocessor.flush())

    assert len(output_chunks) == preprocessor.window_chunks
    assert all(chunk.shape == (512,) for chunk in output_chunks)
    assert all(chunk.dtype == np.float32 for chunk in output_chunks)
    np.testing.assert_allclose(
        np.concatenate(output_chunks),
        np.concatenate(source_chunks) * np.float32(0.5),
    )

    preprocessor._input_queue.put(None)
    preprocessor._processing_thread.join(timeout=2)
    assert not preprocessor._processing_thread.is_alive()


def test_main_pipeline_preprocesses_before_vad():
    source = Path("main.py").read_text(encoding="utf-8")
    capture_loop = source.split("    def _capture_loop(self):", 1)[1].split(
        "    def _enqueue_asr", 1
    )[0]
    vad_helper = source.split("    def _process_vad_input_chunk", 1)[1].split(
        "    def _capture_loop", 1
    )[0]

    assert "self._audio_preprocessor.process_chunk(chunk)" in capture_loop
    assert "self._audio_preprocessor.flush()" in capture_loop
    assert "self._vad.process_chunk(chunk)" not in capture_loop
    assert "self._vad.process_chunk(chunk)" in vad_helper


def test_clearvoice_ready_requires_real_checkpoint_and_runtime(monkeypatch, tmp_path):
    model_root = tmp_path / "models"
    env_root = tmp_path / "envs"
    monkeypatch.setattr(model_manager, "PREPROCESS_MODELS_DIR", model_root)
    monkeypatch.setattr(model_manager, "PREPROCESS_ENVS_DIR", env_root)

    model_dir = model_root / "clearvoice_mossformer2_se"
    model_dir.mkdir(parents=True)
    (model_dir / ".ready").write_text("ready\n", encoding="ascii")
    python = env_root / "clearvoice" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"test")

    # ClearVoice itself may silently continue after a failed download. Our
    # readiness gate must never accept that state without a real checkpoint.
    assert not model_manager.is_audio_preprocessor_ready(
        "clearvoice_mossformer2_se"
    )

    checkpoint = (
        model_dir
        / "checkpoints"
        / "MossFormer2_SE_48K"
        / "last_best_checkpoint.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    with checkpoint.open("wb") as handle:
        handle.seek(100_000_001)
        handle.write(b"\0")

    assert model_manager.is_audio_preprocessor_ready("clearvoice_mossformer2_se")



def test_external_preprocessor_runtime_can_be_reused(monkeypatch, tmp_path):
    shared_env = tmp_path / "shared-demucs"
    python = shared_env / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"stub")

    monkeypatch.setenv("LIVETRANSLATE_DEMUCS_PYTHON", str(shared_env))

    assert model_manager.audio_preprocessor_env_python("demucs_v4") == python
    assert model_manager._ensure_audio_preprocessor_env("demucs_v4") == python


def test_demucs_runtime_reuses_main_torch_without_installing_torch(monkeypatch, tmp_path):
    env_root = tmp_path / "envs"
    env_dir = env_root / "demucs"
    python = env_dir / "Scripts" / "python.exe"
    main_site = tmp_path / "main" / "Lib" / "site-packages"
    main_site.mkdir(parents=True)
    commands = []

    monkeypatch.delenv("LIVETRANSLATE_DEMUCS_PYTHON", raising=False)
    monkeypatch.setattr(model_manager, "PREPROCESS_ENVS_DIR", env_root)
    monkeypatch.setattr(model_manager, "_uv_executable", lambda: "uv")
    monkeypatch.setattr(
        model_manager,
        "_shared_torch_runtime",
        lambda: {
            "python": tmp_path / "main" / "Scripts" / "python.exe",
            "site_packages": main_site,
            "torch": "2.11.0+cu128",
            "torchaudio": "2.11.0+cu128",
            "cuda": "12.8",
        },
    )
    monkeypatch.setattr(
        model_manager,
        "_resolve_shared_overlay_packages",
        lambda *args, **kwargs: [
            "demucs==4.1.0",
            "einops==0.8.2",
            "julius==0.2.8",
            "lameenc==1.8.4",
            "sphn==0.2.1",
        ],
    )

    def fake_run_logged(cmd, **kwargs):
        commands.append((list(cmd), kwargs))
        if len(cmd) > 1 and cmd[1] == "venv":
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_bytes(b"stub")

    monkeypatch.setattr(model_manager, "_run_logged", fake_run_logged)

    assert model_manager._ensure_audio_preprocessor_env("demucs_v4") == python
    assert (env_dir / ".deps-ready").read_text(encoding="ascii") == "shared-main-runtime\n"

    pth = env_dir / "Lib" / "site-packages" / "livetranslate-main-runtime.pth"
    assert pth.is_file()
    assert "main" in pth.read_text(encoding="utf-8")

    install_commands = [cmd for cmd, _ in commands if "install" in cmd]
    assert len(install_commands) == 1
    install_cmd = install_commands[0]
    assert "--no-deps" in install_cmd
    assert "demucs==4.1.0" in install_cmd
    assert "torch==2.8.0" not in install_cmd
    assert "torchaudio==2.8.0" not in install_cmd


def test_demucs_missing_size_uses_shared_runtime_estimate(monkeypatch, tmp_path):
    monkeypatch.delenv("LIVETRANSLATE_DEMUCS_PYTHON", raising=False)
    monkeypatch.setattr(model_manager, "PREPROCESS_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(model_manager, "PREPROCESS_ENVS_DIR", tmp_path / "envs")
    monkeypatch.setattr(
        model_manager,
        "_shared_torch_runtime",
        lambda: {"torch": "2.11.0+cu128"},
    )

    missing = model_manager.get_missing_audio_preprocessor("demucs_v4")
    assert len(missing) == 1
    assert missing[0]["estimated_bytes"] == 85_000_000 + 50_000_000


def test_clearvoice_shared_runtime_keeps_conflicting_packages_in_overlay(monkeypatch, tmp_path):
    env_root = tmp_path / "envs"
    env_dir = env_root / "clearvoice"
    python = env_dir / "Scripts" / "python.exe"
    main_site = tmp_path / "main" / "Lib" / "site-packages"
    main_site.mkdir(parents=True)
    commands = []

    monkeypatch.delenv("LIVETRANSLATE_CLEARVOICE_PYTHON", raising=False)
    monkeypatch.setattr(model_manager, "PREPROCESS_ENVS_DIR", env_root)
    monkeypatch.setattr(model_manager, "_uv_executable", lambda: "uv")
    monkeypatch.setattr(
        model_manager,
        "_shared_torch_runtime",
        lambda: {
            "python": tmp_path / "main" / "Scripts" / "python.exe",
            "site_packages": main_site,
            "torch": "2.11.0+cu128",
            "torchaudio": "2.11.0+cu128",
            "cuda": "12.8",
        },
    )
    monkeypatch.setattr(
        model_manager,
        "_resolve_shared_overlay_packages",
        lambda *args, **kwargs: [
            "clearvoice==0.1.2",
            "numpy==1.26.4",
            "librosa==0.10.2.post1",
            "soundfile==0.12.1",
        ],
    )

    def fake_run_logged(cmd, **kwargs):
        commands.append((list(cmd), kwargs))
        if len(cmd) > 1 and cmd[1] == "venv":
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_bytes(b"stub")

    monkeypatch.setattr(model_manager, "_run_logged", fake_run_logged)

    assert model_manager._ensure_audio_preprocessor_env("clearvoice_mossformer2_se") == python
    install_commands = [cmd for cmd, _ in commands if "install" in cmd]
    assert len(install_commands) == 1
    install_cmd = install_commands[0]
    assert "--no-deps" in install_cmd
    assert "torch==2.8.0" not in install_cmd
    assert "numpy==1.26.4" in install_cmd
    assert "librosa==0.10.2.post1" in install_cmd
    assert "soundfile==0.12.1" in install_cmd


def test_optional_heavy_runtimes_are_uv_managed():
    requirements = Path("requirements.txt").read_text(encoding="utf-8").lower()
    manager = Path("model_manager.py").read_text(encoding="utf-8")

    assert "uv>=0.8,<1.0" in requirements
    assert 'PREPROCESS_ENVS_DIR = APP_DIR / ".preprocess-envs"' in manager
    assert 'packages = ["demucs==4.1.0"]' in manager
    assert '"rnnoise"' not in manager.lower()
    assert "audio-separator" not in manager
    assert '"clearvoice==0.1.2"' in manager
