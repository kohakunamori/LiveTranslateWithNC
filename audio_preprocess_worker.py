from __future__ import annotations

import argparse
import os
import struct
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

# stdout is reserved for the binary request/response protocol. Redirect normal
# print/tqdm output from third-party libraries to stderr before importing them.
_PROTO_OUT = sys.stdout.buffer
sys.stdout = sys.stderr


def _read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        data = stream.read(remaining)
        if not data:
            raise EOFError("audio preprocessing pipe closed")
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)


def _fit_length(samples: np.ndarray, length: int) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=-1)
    samples = samples.reshape(-1)
    if len(samples) > length:
        return samples[:length].copy()
    if len(samples) < length:
        return np.pad(samples, (0, length - len(samples))).astype(np.float32)
    return samples.copy()


class DemucsBackend:
    def __init__(self, model_dir: Path, sample_rate: int):
        import torch
        from demucs.api import Separator

        self.sample_rate = sample_rate
        self.torch = torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.separator = Separator(
            model="htdemucs",
            repo=model_dir.resolve(),
            device=device,
            shifts=1,
            overlap=0.25,
            split=True,
            progress=False,
        )

    def process(self, samples: np.ndarray) -> np.ndarray:
        import torchaudio.functional as AF

        torch = self.torch
        wav = torch.from_numpy(np.asarray(samples, dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():
            _, stems = self.separator.separate_tensor(wav, sr=self.sample_rate)
            vocals = stems["vocals"].mean(dim=0, keepdim=True)
            if self.separator.samplerate != self.sample_rate:
                vocals = AF.resample(
                    vocals,
                    self.separator.samplerate,
                    self.sample_rate,
                )
            output = vocals.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return _fit_length(output, len(samples))

    def close(self):
        try:
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        except Exception:
            pass


class ClearVoiceBackend:
    def __init__(self, model_dir: Path, sample_rate: int):
        self.sample_rate = sample_rate
        self.model_dir = model_dir.resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        # ClearVoice resolves checkpoints/<model> relative to cwd. Keeping cwd in
        # the app's managed model directory makes cache location deterministic.
        os.chdir(self.model_dir)
        from clearvoice import ClearVoice

        self.model = ClearVoice(
            task="speech_enhancement", model_names=["MossFormer2_SE_48K"]
        )
        self._tmp = tempfile.TemporaryDirectory(prefix="livetrans-clearvoice-")
        self.tmp = Path(self._tmp.name)
        self.input_path = self.tmp / "input.wav"
        self.output_path = self.tmp / "output.wav"

    def process(self, samples: np.ndarray) -> np.ndarray:
        import librosa
        import soundfile as sf

        self.output_path.unlink(missing_ok=True)
        sf.write(self.input_path, samples, self.sample_rate, subtype="FLOAT")
        enhanced = self.model(input_path=str(self.input_path), online_write=False)
        self.model.write(enhanced, output_path=str(self.output_path))
        if not self.output_path.exists():
            raise RuntimeError("ClearVoice produced no WAV output")
        audio, sr = sf.read(self.output_path, dtype="float32", always_2d=False)
        if np.ndim(audio) > 1:
            audio = np.asarray(audio, dtype=np.float32).mean(axis=1)
        if sr != self.sample_rate:
            audio = librosa.resample(
                np.asarray(audio, dtype=np.float32),
                orig_sr=sr,
                target_sr=self.sample_rate,
            )
        return _fit_length(audio, len(samples))

    def close(self):
        self._tmp.cleanup()


def _make_backend(mode: str, model_dir: Path, sample_rate: int):
    if mode == "demucs_v4":
        return DemucsBackend(model_dir, sample_rate)
    if mode == "clearvoice_mossformer2_se":
        return ClearVoiceBackend(model_dir, sample_rate)
    raise ValueError(f"Unsupported audio preprocessing mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    backend = _make_backend(args.mode, Path(args.model_dir), args.sample_rate)
    _PROTO_OUT.write(b"READY\n")
    _PROTO_OUT.flush()

    try:
        while True:
            header = sys.stdin.buffer.read(4)
            if not header:
                break
            if len(header) != 4:
                raise EOFError("short preprocessing request header")
            count = struct.unpack("<I", header)[0]
            if count == 0:
                break
            payload = _read_exact(sys.stdin.buffer, count * 4)
            samples = np.frombuffer(payload, dtype=np.float32).copy()
            try:
                output = backend.process(samples)
            except Exception:
                traceback.print_exc(file=sys.stderr)
                _PROTO_OUT.write(struct.pack("<I", 0))
                _PROTO_OUT.flush()
                return 2
            output = _fit_length(output, count)
            _PROTO_OUT.write(struct.pack("<I", len(output)))
            _PROTO_OUT.write(output.astype(np.float32, copy=False).tobytes())
            _PROTO_OUT.flush()
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
