from pathlib import Path

import model_download_worker


def test_download_worker_routes_preprocessor_to_model_manager(monkeypatch):
    calls = []
    monkeypatch.setattr(
        model_download_worker,
        "download_audio_preprocessor",
        lambda mode, proxy="system": calls.append((mode, proxy)),
    )

    model_download_worker._download_one(
        {"name": "Demucs v4", "type": "preprocess:demucs_v4"},
        hub="hf",
        proxy="http://127.0.0.1:7890",
    )

    assert calls == [("demucs_v4", "http://127.0.0.1:7890")]


def test_download_worker_routes_funasr_model_key(monkeypatch):
    calls = []

    def fake_download(engine, **kwargs):
        calls.append((engine, kwargs))

    monkeypatch.setattr(model_download_worker, "download_asr", fake_download)
    model_download_worker._download_one(
        {"name": "SenseVoice Small", "type": "funasr:sensevoice-small"},
        hub="ms",
        proxy="system",
    )

    assert calls == [
        (
            "funasr",
            {
                "model_size": "sensevoice-small",
                "hub": "ms",
                "proxy": "system",
            },
        )
    ]


def test_model_download_dialog_uses_isolated_process_without_global_log_capture():
    source = Path("dialogs.py").read_text(encoding="utf-8")
    model_dialog = source.split("class ModelDownloadDialog", 1)[1].split(
        "class ModelEditDialog", 1
    )[0]
    setup_dialog = source.split("class SetupWizardDialog", 1)[1].split(
        "class ModelDownloadDialog", 1
    )[0]

    for section in (model_dialog, setup_dialog):
        assert "_IsolatedDownloadProcess" in section
        assert "threading.Thread" not in section
        assert "sys.stderr" not in section
        assert "logging.getLogger().addHandler" not in section


def test_download_worker_has_dedicated_non_propagating_log():
    source = Path("model_download_worker.py").read_text(encoding="utf-8")

    assert 'DOWNLOAD_LOGGER_NAME = "LiveTranslate.Download"' in source
    assert 'log_dir = APP_DIR / "logs" / "downloads"' in source
    assert "logger.propagate = False" in source
    assert 'logging.StreamHandler(sys.stdout)' in source
