from pathlib import Path
import threading

import numpy as np
import yaml

import model_manager
from audio_preprocess_worker import RNNoiseBackend
from audio_preprocessor import AudioPreprocessor


def test_audio_preprocessing_defaults_to_off():
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))

    assert config["audio"]["preprocess_mode"] == "off"
    assert model_manager.normalize_audio_preprocess_mode(None) == "off"
    assert model_manager.normalize_audio_preprocess_mode("unknown") == "off"


def test_supported_audio_preprocessing_modes_are_registered():
    assert set(model_manager.PREPROCESSOR_PROFILES) == {
        "off",
        "rnnoise",
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


def test_buffered_preprocessor_returns_original_32ms_chunk_shape(monkeypatch):
    """Exercise buffering/async scheduling without loading an external model."""
    preprocessor = AudioPreprocessor(
        "rnnoise", sample_rate=16000, chunk_duration=0.032
    )
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


def test_rnnoise_ready_requires_model_and_arnndn_runtime(monkeypatch, tmp_path):
    model_root = tmp_path / "models"
    monkeypatch.setattr(model_manager, "PREPROCESS_MODELS_DIR", model_root)
    model_dir = model_root / "rnnoise"
    model_dir.mkdir(parents=True)
    (model_dir / ".ready").write_text("ready\n", encoding="ascii")
    model = model_dir / "std.rnnn"
    model.write_bytes(b"x" * 100_001)

    monkeypatch.setattr(model_manager, "audio_preprocessor_ffmpeg_executable", lambda: None)
    assert not model_manager.is_audio_preprocessor_ready("rnnoise")

    fake_ffmpeg = model_dir / "runtime" / "ffmpeg.exe"
    fake_ffmpeg.parent.mkdir(parents=True)
    fake_ffmpeg.write_bytes(b"fake")
    monkeypatch.setattr(
        model_manager, "audio_preprocessor_ffmpeg_executable", lambda: fake_ffmpeg
    )
    monkeypatch.setattr(model_manager, "_ffmpeg_supports_arnndn", lambda exe: True)
    assert model_manager.is_audio_preprocessor_ready("rnnoise")


def test_rnnoise_worker_prefers_managed_ffmpeg(tmp_path, monkeypatch):
    model = tmp_path / "std.rnnn"
    model.write_bytes(b"x")
    managed = tmp_path / "runtime" / "ffmpeg.exe"
    managed.parent.mkdir(parents=True)
    managed.write_bytes(b"fake")
    monkeypatch.setattr("audio_preprocess_worker.shutil.which", lambda name: "PATH-ffmpeg.exe")

    backend = RNNoiseBackend(tmp_path, 16000)

    assert backend.ffmpeg == str(managed.resolve())


def test_rnnoise_managed_runtime_installer_extracts_imageio_binary(
    tmp_path, monkeypatch
):
    model_dir = tmp_path / "rnnoise"
    monkeypatch.setattr(model_manager, "APP_DIR", tmp_path)
    monkeypatch.setattr(model_manager, "audio_preprocessor_ffmpeg_executable", lambda: None)
    monkeypatch.setattr(model_manager, "_uv_executable", lambda: "uv")
    monkeypatch.setattr(model_manager, "_ffmpeg_supports_arnndn", lambda exe: True)

    def fake_run_logged(cmd, **kwargs):
        target = Path(cmd[cmd.index("--target") + 1])
        binary = target / "imageio_ffmpeg" / "binaries" / "ffmpeg-win-x86_64-v7.1.exe"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"managed-ffmpeg")

    monkeypatch.setattr(model_manager, "_run_logged", fake_run_logged)

    installed = model_manager._ensure_rnnoise_ffmpeg(
        model_manager.PREPROCESSOR_PROFILES["rnnoise"], model_dir
    )

    assert installed == model_dir / "runtime" / "ffmpeg.exe"
    assert installed.read_bytes() == b"managed-ffmpeg"
    assert not any((tmp_path / ".tmp").glob("rnnoise-ffmpeg-*"))


def test_optional_heavy_runtimes_are_uv_managed():
    requirements = Path("requirements.txt").read_text(encoding="utf-8").lower()
    manager = Path("model_manager.py").read_text(encoding="utf-8")

    assert "uv>=0.8,<1.0" in requirements
    assert 'PREPROCESS_ENVS_DIR = APP_DIR / ".preprocess-envs"' in manager
    assert '"ffmpeg_package": "imageio-ffmpeg==0.6.0"' in manager
    assert 'packages = ["demucs==4.1.0"]' in manager
    assert "audio-separator" not in manager
    assert '"clearvoice==0.1.2"' in manager
