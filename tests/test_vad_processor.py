import sys
import types
from pathlib import Path

import numpy as np
import torch

from vad_processor import VADProcessor


class FakeSileroModel:
    def __init__(self):
        self.reset_count = 0
        self.calls_since_reset = 0

    def eval(self):
        return self

    def reset_states(self):
        self.reset_count += 1
        self.calls_since_reset = 0

    def __call__(self, tensor, sample_rate):
        assert sample_rate == 16000
        assert tensor.numel() == 512
        self.calls_since_reset += 1
        # Deterministic speech probability based on signal energy while still
        # behaving like a stateful recurrent model for reset accounting.
        probability = 0.9 if float(tensor.abs().mean()) > 0.01 else 0.0
        return torch.tensor(probability)


def make_vad(monkeypatch, model=None, *, min_speech_duration=0.064):
    model = model or FakeSileroModel()
    module = types.ModuleType("silero_vad")
    module.load_silero_vad = lambda: model
    monkeypatch.setitem(sys.modules, "silero_vad", module)
    vad = VADProcessor(
        sample_rate=16000,
        threshold=0.5,
        min_speech_duration=min_speech_duration,
        max_speech_duration=8.0,
        chunk_duration=0.032,
    )
    vad.update_settings({"silence_mode": "fixed", "silence_duration": 0.064})
    return vad, model


def test_silero_state_resets_when_segment_ends(monkeypatch):
    vad, model = make_vad(monkeypatch)
    speech = np.full(512, 0.1, dtype=np.float32)
    silence = np.zeros(512, dtype=np.float32)

    assert vad.process_chunk(speech) is None
    assert vad.process_chunk(speech) is None
    assert vad.process_chunk(silence) is None
    segment = vad.process_chunk(silence)

    assert segment is not None
    assert len(segment) == 4 * 512
    assert model.reset_count == 1
    assert model.calls_since_reset == 0
    assert not vad._is_speaking


def test_vad_mode_change_starts_fresh_silero_stream(monkeypatch):
    vad, model = make_vad(monkeypatch)
    speech = np.full(512, 0.1, dtype=np.float32)

    vad.process_chunk(speech)
    assert vad._is_speaking
    assert model.calls_since_reset == 1

    vad.update_settings({"vad_mode": "energy"})
    assert vad.mode == "energy"
    assert not vad._is_speaking
    assert model.reset_count == 1
    assert model.calls_since_reset == 0

    vad.update_settings({"vad_mode": "silero"})
    assert vad.mode == "silero"
    assert model.reset_count == 2


def test_hard_reset_clears_prespeech_and_model_state(monkeypatch):
    vad, model = make_vad(monkeypatch)
    silence = np.zeros(512, dtype=np.float32)

    for _ in range(3):
        vad.process_chunk(silence)
    assert len(vad._pre_buffer) == 3
    assert model.calls_since_reset == 3

    vad._reset()

    assert len(vad._pre_buffer) == 0
    assert model.reset_count == 1
    assert model.calls_since_reset == 0


def test_pause_resets_vad_stream_under_lock():
    source = Path("main.py").read_text(encoding="utf-8")
    pause_body = source.split("    def pause(self):", 1)[1].split(
        "    def resume(self):", 1
    )[0]

    assert "with self._audio_preprocess_lock:" in pause_body
    assert "self._audio_preprocessor.reset()" in pause_body
    assert "with self._vad_lock:" in pause_body
    assert "self._vad._reset()" in pause_body
