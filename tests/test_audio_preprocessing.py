from pathlib import Path
import threading

import numpy as np
import pytest
import yaml

import audio_preprocess_worker
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
        "mdx_net",
        "melband_roformer",
    }
    assert (
        model_manager.PREPROCESSOR_PROFILES["mdx_net"]["model_filename"]
        == "UVR-MDX-NET-Inst_HQ_3.onnx"
    )
    assert (
        model_manager.PREPROCESSOR_PROFILES["melband_roformer"]["model_filename"]
        == "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"
    )


def test_live_separator_modes_share_runtime_and_model_cache():
    assert model_manager.audio_preprocessor_model_dir(
        "mdx_net"
    ) == model_manager.audio_preprocessor_model_dir("melband_roformer")
    assert model_manager.audio_preprocessor_env_python(
        "mdx_net"
    ) == model_manager.audio_preprocessor_env_python("melband_roformer")
    assert model_manager.audio_preprocessor_ready_marker(
        "mdx_net"
    ) != model_manager.audio_preprocessor_ready_marker("melband_roformer")


def test_off_mode_preserves_capture_pcm_contract():
    chunk = np.linspace(-0.5, 0.5, 512, dtype=np.float32)
    preprocessor = AudioPreprocessor("off", sample_rate=16000, chunk_duration=0.032)

    output = preprocessor.process_chunk(chunk)

    assert len(output) == 1
    np.testing.assert_array_equal(output[0], chunk)
    assert output[0].dtype == np.float32


@pytest.mark.parametrize("mode", ["mdx_net", "melband_roformer"])
def test_streaming_trailing_window_preserves_pcm_timeline(monkeypatch, mode):
    """Exercise live trailing-window scheduling without loading an external model."""
    preprocessor = AudioPreprocessor(mode, sample_rate=16000, chunk_duration=0.032)
    monkeypatch.setattr(preprocessor, "start", lambda: None)
    monkeypatch.setattr(
        preprocessor,
        "_request",
        lambda samples, *, input_rate, output_samples: np.asarray(
            samples, dtype=np.float32
        )[:, 0][::3][:output_samples]
        * np.float32(0.5),
    )
    preprocessor._processing_thread = threading.Thread(
        target=preprocessor._processing_loop, daemon=True
    )
    preprocessor._processing_thread.start()

    source_count = preprocessor.window_chunks + preprocessor.hop_chunks + 5
    source_chunks = [
        np.full((1536, 2), (i + 1) / 100.0, dtype=np.float32)
        for i in range(source_count)
    ]
    mic_chunks = [np.full(512, 0.01, dtype=np.float32) for _ in range(source_count)]
    output_chunks = []
    for chunk, mic in zip(source_chunks, mic_chunks):
        output_chunks.extend(
            preprocessor.process_chunk(chunk, input_rate=48000, mic_chunk=mic)
        )
    output_chunks.extend(preprocessor.flush())

    assert len(output_chunks) == len(source_chunks)
    assert all(chunk.shape == (512,) for chunk in output_chunks)
    assert all(chunk.dtype == np.float32 for chunk in output_chunks)
    np.testing.assert_allclose(
        np.concatenate(output_chunks),
        np.concatenate(
            [np.full(512, (i + 1) / 200.0 + 0.01, dtype=np.float32) for i in range(source_count)]
        ),
        atol=1e-7,
    )

    preprocessor._input_queue.put(None)
    preprocessor._processing_thread.join(timeout=2)
    assert not preprocessor._processing_thread.is_alive()


def test_short_stream_flushes_without_waiting_for_full_window(monkeypatch):
    preprocessor = AudioPreprocessor("mdx_net", sample_rate=16000, chunk_duration=0.032)
    monkeypatch.setattr(preprocessor, "start", lambda: None)
    monkeypatch.setattr(
        preprocessor,
        "_request",
        lambda samples, *, input_rate, output_samples: np.asarray(
            samples, dtype=np.float32
        )[:, 0][::3][:output_samples],
    )
    preprocessor._processing_thread = threading.Thread(
        target=preprocessor._processing_loop, daemon=True
    )
    preprocessor._processing_thread.start()

    chunks = [np.full((1536, 2), i + 1, dtype=np.float32) for i in range(7)]
    for chunk in chunks:
        assert preprocessor.process_chunk(chunk, input_rate=48000) == []
    output = preprocessor.flush()

    assert len(output) == len(chunks)
    np.testing.assert_array_equal(
        np.concatenate(output),
        np.concatenate([np.full(512, i + 1, dtype=np.float32) for i in range(7)]),
    )

    preprocessor._input_queue.put(None)
    preprocessor._processing_thread.join(timeout=2)


def test_trailing_scheduler_emits_before_full_model_window(monkeypatch):
    preprocessor = AudioPreprocessor("mdx_net", sample_rate=16000, chunk_duration=0.032)
    monkeypatch.setattr(preprocessor, "start", lambda: None)
    scheduled = []

    def capture_enqueue(block, input_rate, mic_block, output_start, output_end, **kwargs):
        scheduled.append((block, mic_block, output_start, output_end))

    monkeypatch.setattr(preprocessor, "_enqueue_block", capture_enqueue)
    needed = preprocessor.hop_chunks + preprocessor.lookahead_chunks
    assert needed < preprocessor.window_chunks

    for i in range(needed - 1):
        chunk = np.full((1536, 2), i + 1, dtype=np.float32)
        preprocessor.process_chunk(chunk, input_rate=48000)
        assert scheduled == []

    final = np.full((1536, 2), needed, dtype=np.float32)
    preprocessor.process_chunk(final, input_rate=48000)

    assert len(scheduled) == 1
    block, mic_block, output_start, output_end = scheduled[0]
    assert block.shape == (preprocessor.window_chunks * 1536, 2)
    assert mic_block.shape == (preprocessor.window_samples,)
    assert output_start == preprocessor.history_samples
    assert output_end - output_start == preprocessor.hop_samples
    # Startup gets left-padded history while retaining the full model context.
    assert np.all(block[: preprocessor.history_chunks * 1536] == 0)
    assert preprocessor.latency_seconds == pytest.approx(needed * 0.032)
    assert preprocessor.latency_seconds < preprocessor.model_window_seconds


@pytest.mark.parametrize(
    ("mode", "measured_seconds", "expected_min"),
    [
        ("mdx_net", 0.20, 12),
        ("mdx_net", 0.40, 20),
        ("melband_roformer", 0.20, 16),
        ("melband_roformer", 0.40, 21),
    ],
)
def test_adaptive_hop_keeps_gpu_headroom(mode, measured_seconds, expected_min):
    preprocessor = AudioPreprocessor(mode, sample_rate=16000, chunk_duration=0.032)
    required = int(
        np.ceil(
            measured_seconds
            / (preprocessor.target_hop_rtf * preprocessor.chunk_duration)
        )
    )
    preprocessor._adapt_hop_from_benchmark(measured_seconds)

    assert preprocessor._benchmark_inference_seconds == pytest.approx(measured_seconds)
    assert preprocessor.hop_chunks >= expected_min
    assert preprocessor.hop_chunks >= required
    assert preprocessor.hop_chunks + preprocessor.lookahead_chunks <= preprocessor.window_chunks
    assert (
        measured_seconds / (preprocessor.hop_chunks * preprocessor.chunk_duration)
        <= preprocessor.target_hop_rtf + 1e-9
    )
    assert preprocessor.estimated_total_latency_seconds < preprocessor.model_window_seconds


def test_main_pipeline_preprocesses_before_vad():
    source = Path("main.py").read_text(encoding="utf-8")
    capture_loop = source.split("    def _capture_loop(self):", 1)[1].split(
        "    def _enqueue_asr", 1
    )[0]
    vad_helper = source.split("    def _process_vad_input_chunk", 1)[1].split(
        "    def _capture_loop", 1
    )[0]

    assert "self._audio_preprocessor.process_chunk(chunk)" in capture_loop
    assert "item.loopback_native" in capture_loop
    assert "input_rate=item.loopback_rate" in capture_loop
    assert "mic_chunk=item.mic_mono" in capture_loop
    assert "self._audio_preprocessor.flush()" in capture_loop
    assert "self._vad.process_chunk(chunk)" not in capture_loop
    assert "self._vad.process_chunk(chunk)" in vad_helper


def test_live_models_require_real_model_files_and_shared_runtime(monkeypatch, tmp_path):
    model_root = tmp_path / "models"
    env_root = tmp_path / "envs"
    monkeypatch.setattr(model_manager, "PREPROCESS_MODELS_DIR", model_root)
    monkeypatch.setattr(model_manager, "PREPROCESS_ENVS_DIR", env_root)
    monkeypatch.setitem(
        model_manager.PREPROCESSOR_PROFILES["mdx_net"], "min_model_bytes", 8
    )
    monkeypatch.setitem(
        model_manager.PREPROCESSOR_PROFILES["melband_roformer"],
        "min_model_bytes",
        8,
    )

    model_dir = model_root / "audio-separator-live"
    model_dir.mkdir(parents=True)
    python = env_root / "audio-separator-live" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"test")

    mdx_marker = model_manager.audio_preprocessor_ready_marker("mdx_net")
    mdx_marker.write_text("ready\n", encoding="ascii")
    assert not model_manager.is_audio_preprocessor_ready("mdx_net")
    (model_dir / "UVR-MDX-NET-Inst_HQ_3.onnx").write_bytes(b"123456789")
    assert not model_manager.is_audio_preprocessor_ready("mdx_net")
    (model_dir / "mdx_model_data.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "vr_model_data.json").write_text("{}\n", encoding="utf-8")
    assert model_manager.is_audio_preprocessor_ready("mdx_net")

    mel_marker = model_manager.audio_preprocessor_ready_marker("melband_roformer")
    mel_marker.write_text("ready\n", encoding="ascii")
    assert not model_manager.is_audio_preprocessor_ready("melband_roformer")
    (model_dir / "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt").write_bytes(b"123456789")
    (model_dir / "model_mel_band_roformer_ep_3005_sdr_11.4360.yaml").write_text(
        "audio: {}\n", encoding="utf-8"
    )
    assert model_manager.is_audio_preprocessor_ready("melband_roformer")


def test_external_audio_separator_runtime_can_be_reused_by_both_modes(
    monkeypatch, tmp_path
):
    shared_env = tmp_path / "shared-audio-separator"
    python = shared_env / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"stub")

    monkeypatch.setenv("LIVETRANSLATE_AUDIO_SEPARATOR_PYTHON", str(shared_env))

    assert model_manager.audio_preprocessor_env_python("mdx_net") == python
    assert model_manager.audio_preprocessor_env_python("melband_roformer") == python
    assert model_manager._ensure_audio_preprocessor_env("mdx_net") == python
    assert model_manager._ensure_audio_preprocessor_env("melband_roformer") == python


def test_audio_separator_runtime_reuses_main_torch_without_installing_torch(
    monkeypatch, tmp_path
):
    env_root = tmp_path / "envs"
    env_dir = env_root / "audio-separator-live"
    python = env_dir / "Scripts" / "python.exe"
    main_site = tmp_path / "main" / "Lib" / "site-packages"
    main_site.mkdir(parents=True)
    commands = []

    monkeypatch.delenv("LIVETRANSLATE_AUDIO_SEPARATOR_PYTHON", raising=False)
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
            "cuda": "cu128",
        },
    )
    monkeypatch.setattr(
        model_manager,
        "_resolve_shared_overlay_packages",
        lambda *args, **kwargs: [
            "audio-separator==0.47.0",
            "onnxruntime-gpu==1.23.0",
            "numpy==2.3.0",
        ],
    )

    def fake_run_logged(cmd, **kwargs):
        commands.append((list(cmd), kwargs))
        if len(cmd) > 1 and cmd[1] == "venv":
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_bytes(b"stub")

    monkeypatch.setattr(model_manager, "_run_logged", fake_run_logged)

    assert model_manager._ensure_audio_preprocessor_env("mdx_net") == python
    assert (env_dir / ".deps-ready").read_text(encoding="ascii") == "shared-main-runtime\n"
    assert model_manager._ensure_audio_preprocessor_env("melband_roformer") == python

    pth = env_dir / "Lib" / "site-packages" / "livetranslate-main-runtime.pth"
    assert pth.is_file()
    assert "main" in pth.read_text(encoding="utf-8")

    install_commands = [cmd for cmd, _ in commands if "install" in cmd]
    assert len(install_commands) == 1
    install_cmd = install_commands[0]
    assert "--no-deps" in install_cmd
    assert "audio-separator==0.47.0" in install_cmd
    assert not any(arg.startswith("torch==") for arg in install_cmd)
    assert not any(arg.startswith("torchaudio==") for arg in install_cmd)


def test_missing_size_uses_shared_runtime_estimate(monkeypatch, tmp_path):
    monkeypatch.delenv("LIVETRANSLATE_AUDIO_SEPARATOR_PYTHON", raising=False)
    monkeypatch.setattr(model_manager, "PREPROCESS_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(model_manager, "PREPROCESS_ENVS_DIR", tmp_path / "envs")
    monkeypatch.setattr(
        model_manager,
        "_shared_torch_runtime",
        lambda: {"torch": "2.11.0+cu128"},
    )

    missing = model_manager.get_missing_audio_preprocessor("mdx_net")
    profile = model_manager.PREPROCESSOR_PROFILES["mdx_net"]
    assert len(missing) == 1
    assert missing[0]["estimated_bytes"] == (
        profile["estimated_bytes"] + profile["shared_runtime_estimated_bytes"]
    )


def test_mdx_primary_instrumental_is_converted_to_vocals_residual():
    class FakeMdx:
        primary_stem_name = "Instrumental"
        compensate = 1.0

        def demix(self, mix):
            return mix * np.float32(0.75)

    mix = np.full((2, 100), 0.8, dtype=np.float32)
    vocals = audio_preprocess_worker.AudioSeparatorLiveBackend._vocals_from_demix(
        FakeMdx(), mix
    )
    np.testing.assert_allclose(vocals, mix * np.float32(0.25))


def test_roformer_vocals_dict_is_selected_directly():
    class FakeRoformer:
        def demix(self, mix):
            return {
                "vocals": np.full_like(mix, 0.2),
                "other": np.full_like(mix, 0.8),
            }

    mix = np.ones((2, 100), dtype=np.float32)
    vocals = audio_preprocess_worker.AudioSeparatorLiveBackend._vocals_from_demix(
        FakeRoformer(), mix
    )
    np.testing.assert_allclose(vocals, 0.2)


def test_worker_preserves_native_stereo_until_demix():
    class FakeModel:
        primary_stem_name = "vocals"

        def __init__(self):
            self.seen = None

        def demix(self, mix):
            self.seen = np.asarray(mix, dtype=np.float32).copy()
            return self.seen

    backend = object.__new__(audio_preprocess_worker.AudioSeparatorLiveBackend)
    backend.target_sample_rate = 44100
    backend.model = FakeModel()

    native = np.column_stack(
        (
            np.full(256, 0.1, dtype=np.float32),
            np.full(256, 0.3, dtype=np.float32),
        )
    )
    output = backend.process(native, input_rate=44100, output_samples=256)

    assert backend.model.seen.shape == (2, 256)
    np.testing.assert_allclose(backend.model.seen[0], 0.1)
    np.testing.assert_allclose(backend.model.seen[1], 0.3)
    np.testing.assert_allclose(output, 0.2)


def test_markerless_partial_model_is_replaced_before_download(monkeypatch, tmp_path):
    model_root = tmp_path / "models"
    env_root = tmp_path / "envs"
    model_dir = model_root / "audio-separator-live"
    model_dir.mkdir(parents=True)
    target = model_dir / "UVR-MDX-NET-Inst_HQ_3.onnx"
    target.write_bytes(b"partial-data")

    fake_python = env_root / "audio-separator-live" / "Scripts" / "python.exe"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_bytes(b"stub")

    monkeypatch.setattr(model_manager, "PREPROCESS_MODELS_DIR", model_root)
    monkeypatch.setattr(model_manager, "PREPROCESS_ENVS_DIR", env_root)
    monkeypatch.setitem(
        model_manager.PREPROCESSOR_PROFILES["mdx_net"], "min_model_bytes", 8
    )
    monkeypatch.setattr(
        model_manager, "_ensure_audio_preprocessor_env", lambda mode: fake_python
    )
    monkeypatch.setattr(
        model_manager, "audio_preprocessor_env_python", lambda mode: fake_python
    )

    def fake_run_logged(cmd, **kwargs):
        assert not target.exists(), "markerless partial must be deleted first"
        target.write_bytes(b"complete-model")
        (model_dir / "mdx_model_data.json").write_text("{}\n", encoding="utf-8")
        (model_dir / "vr_model_data.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(model_manager, "_run_logged", fake_run_logged)
    monkeypatch.setattr(model_manager.shutil, "rmtree", lambda *args, **kwargs: None)

    model_manager.download_audio_preprocessor("mdx_net")

    assert target.read_bytes() == b"complete-model"
    assert model_manager.audio_preprocessor_ready_marker("mdx_net").is_file()


def test_optional_live_separator_runtime_is_uv_managed():
    requirements = Path("requirements.txt").read_text(encoding="utf-8").lower()
    manager = Path("model_manager.py").read_text(encoding="utf-8")
    worker = Path("audio_preprocess_worker.py").read_text(encoding="utf-8")

    assert "uv>=0.8,<1.0" in requirements
    assert 'PREPROCESS_ENVS_DIR = APP_DIR / ".preprocess-envs"' in manager
    assert '"audio-separator[gpu]==0.47.0"' in manager
    assert '"onnxruntime-gpu>=1.21,<1.27"' in manager
    assert '"segment_size": 256' in worker
    assert '"overlap": 0.0' in worker
    assert '"cudnn_conv_algo_search": "HEURISTIC"' in worker
    assert "download_model_files = _local_model_files" in worker
    assert "actual_ort_providers" in worker
    assert "OrtValue.from_dlpack" in worker
    assert "run_with_iobinding" in worker
    assert "_enable_mdx_cuda_iobinding" in worker
    dependency_lines = [
        line.strip().lower()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert not any(line.startswith("audio-separator") for line in dependency_lines)
    assert '"rnnoise"' not in manager.lower()
