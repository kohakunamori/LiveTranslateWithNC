from __future__ import annotations

"""Isolated live music-separation worker.

The in-memory ``demix`` streaming path is adapted from Nicholas N.'s
``python-audio-separator-live`` (MIT), while capture/VAD scheduling stays owned
by LiveTranslate.  We deliberately avoid the file-based ``separate()`` API so a
loaded MDX-NET or MelBand RoFormer model can process overlapping live windows.
"""

import argparse
import logging
import math
import struct
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np

# stdout is reserved for the binary request/response protocol. Redirect normal
# print/tqdm output from third-party libraries to stderr before importing them.
_PROTO_OUT = sys.stdout.buffer
sys.stdout = sys.stderr

# Third-party compatibility warnings from rotary-embedding-torch/pydub are
# non-actionable for the live worker and otherwise pollute LiveTranslate's
# runtime log once per process start.
warnings.filterwarnings(
    "ignore", category=FutureWarning, module=r"rotary_embedding_torch(\..*)?"
)
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"pydub(\..*)?")

MODEL_FILENAMES = {
    "mdx_net": "UVR-MDX-NET-Inst_HQ_3.onnx",
    "melband_roformer": "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
}
MODEL_SAMPLE_RATE = 44100


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
        # Model output is stereo/channel-first.  ASR/VAD consume mono.
        axis = 0 if samples.shape[0] <= 8 else -1
        samples = samples.mean(axis=axis)
    samples = samples.reshape(-1)
    if len(samples) > length:
        return samples[:length].copy()
    if len(samples) < length:
        return np.pad(samples, (0, length - len(samples))).astype(np.float32)
    return samples.copy()


def _resample(samples: np.ndarray, source_rate: int, target_rate: int, *, axis: int = -1) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if source_rate == target_rate:
        return samples.copy()
    from scipy.signal import resample_poly

    divisor = math.gcd(int(source_rate), int(target_rate))
    output = resample_poly(
        samples,
        int(target_rate) // divisor,
        int(source_rate) // divisor,
        axis=axis,
    )
    return np.asarray(output, dtype=np.float32)


class AudioSeparatorLiveBackend:
    """Persistent in-memory MDX/RoFormer separator for overlapping live windows."""

    def __init__(
        self,
        mode: str,
        model_dir: Path,
        target_sample_rate: int,
        window_seconds: float,
    ):
        import torch
        from audio_separator.separator import Separator
        from audio_separator.separator.architectures import mdx_separator, mdxc_separator

        # audio-separator emits one tqdm bar per demix window by default; that is
        # useful for files but pure log noise in a live pipeline.
        mdx_separator.tqdm = mdxc_separator.tqdm = lambda it, **kw: it

        if mode not in MODEL_FILENAMES:
            raise ValueError(f"Unsupported live separator mode: {mode}")
        self.mode = mode
        self.target_sample_rate = int(target_sample_rate)
        self.window_seconds = float(window_seconds)
        self.model_dir = model_dir.resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.torch = torch

        use_native_fp16 = mode == "melband_roformer" and torch.cuda.is_available()
        self.separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=str(self.model_dir),
            sample_rate=MODEL_SAMPLE_RATE,
            use_native_fp16=use_native_fp16,
            mdx_params={
                "hop_length": 1024,
                # mdx_dim_t_set=8 means an actual native dim_t of 2**8=256.
                # Keep that native size so audio-separator stays on ONNX Runtime
                # instead of converting the graph to PyTorch. The outer live
                # scheduler already overlaps/trims window edges, so a second
                # internal MDX overlap would only duplicate expensive inference.
                "segment_size": 256,
                "overlap": 0.0,
                "batch_size": 1,
                "enable_denoise": False,
            },
            mdxc_params={
                "segment_size": 256,
                "override_model_segment_size": False,
                "batch_size": 1,
                "overlap": None,
                "pitch_shift": 0,
            },
        )
        if mode == "mdx_net" and torch.cuda.is_available():
            # ORT's default CUDA cuDNN convolution algorithm search is
            # exhaustive and made the first live MDX inference take ~40 s on
            # the RTX 5070 Laptop. Heuristic search avoids that one-time stall
            # while preserving the steady-state CUDA EP path.
            self.separator.onnx_execution_provider = [
                (
                    "CUDAExecutionProvider",
                    {"cudnn_conv_algo_search": "HEURISTIC"},
                )
            ]
        model_filename = MODEL_FILENAMES[mode]
        model_path = self.model_dir / model_filename
        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        # LiveTranslate owns model download/readiness. Avoid audio-separator's
        # load-time list_supported_model_files() network call, which otherwise
        # makes every mode activation depend on GitHub even with complete local
        # weights. Return the already-managed local model/config directly.
        if mode == "mdx_net":
            def _local_model_files(_filename):
                return (
                    model_filename,
                    "MDX",
                    "UVR-MDX-NET Inst HQ 3",
                    str(model_path),
                    None,
                )
        else:
            yaml_name = model_path.with_suffix(".yaml").name
            yaml_path = self.model_dir / yaml_name
            if not yaml_path.is_file():
                raise FileNotFoundError(yaml_path)

            def _local_model_files(_filename):
                return (
                    model_filename,
                    "MDXC",
                    "MelBand RoFormer ep3005 SDR 11.4360",
                    str(model_path),
                    yaml_name,
                )

        self.separator.download_model_files = _local_model_files
        # audio-separator 0.47 compares a provider tuple (provider, options)
        # directly with the string list returned by ORT and therefore emits a
        # false "could not activate" warning for our heuristic CUDA provider.
        # Suppress that one load-time warning and validate the actual session
        # ourselves immediately afterwards.
        previous_logger_level = self.separator.logger.level
        if mode == "mdx_net":
            self.separator.logger.setLevel(logging.ERROR)
        try:
            self.separator.load_model(model_filename)
        finally:
            self.separator.logger.setLevel(previous_logger_level)
        self.model = self.separator.model_instance

        actual_ort_providers = None
        if mode == "mdx_net":
            model_run = getattr(self.model, "model_run", None)
            for cell in getattr(model_run, "__closure__", None) or ():
                try:
                    candidate = cell.cell_contents
                except ValueError:
                    continue
                if hasattr(candidate, "get_providers"):
                    actual_ort_providers = list(candidate.get_providers())
                    break
            if torch.cuda.is_available() and (
                not actual_ort_providers
                or "CUDAExecutionProvider" not in actual_ort_providers
            ):
                raise RuntimeError(
                    "MDX-NET ONNX session did not activate CUDAExecutionProvider; "
                    f"actual providers={actual_ort_providers}"
                )

        # RoFormer models normally use an ~8 s inference segment.  For live
        # separation, match that internal segment to our ~1.6 s outer window.
        # This is the key optimization used by python-audio-separator-live.
        if getattr(self.model, "is_roformer", False):
            cfg = self.model.model_data_cfgdict
            hop = int(
                getattr(cfg.model, "stft_hop_length", None)
                or cfg.audio.hop_length
            )
            model_window_samples = round(self.window_seconds * MODEL_SAMPLE_RATE)
            self.model.override_model_segment_size = True
            self.model.segment_size = max(2, model_window_samples // hop + 1)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ort_provider = "n/a"
        try:
            import onnxruntime as ort

            providers = actual_ort_providers or ort.get_available_providers()
            ort_provider = (
                "CUDAExecutionProvider"
                if "CUDAExecutionProvider" in providers
                else providers[0] if providers else "none"
            )
        except Exception:
            pass
        print(
            f"live separator loaded: mode={mode}, model={model_filename}, "
            f"torch_device={device}, ort_provider={ort_provider}, "
            f"window={self.window_seconds:.3f}s"
        )
        if mode == "mdx_net" and torch.cuda.is_available() and ort_provider != "CUDAExecutionProvider":
            print(
                "WARNING: MDX-NET is not using CUDAExecutionProvider; realtime "
                "performance may be insufficient",
                file=sys.stderr,
            )

    @staticmethod
    def _vocals_from_demix(model, mix: np.ndarray) -> np.ndarray:
        separated = model.demix(mix)
        if isinstance(separated, dict):
            key = next(
                (name for name in separated if str(name).lower() == "vocals"),
                None,
            )
            if key is None:
                raise RuntimeError(
                    f"separator returned no vocals stem: {list(separated)}"
                )
            vocals = np.asarray(separated[key], dtype=np.float32)
            while vocals.ndim > 2 and vocals.shape[0] == 1:
                vocals = vocals[0]
            return vocals

        primary = np.asarray(separated, dtype=np.float32)
        while primary.ndim > 2 and primary.shape[0] == 1:
            primary = primary[0]
        primary_name = str(getattr(model, "primary_stem_name", ""))
        if primary_name.lower() == "vocals":
            return primary

        # UVR-MDX-NET-Inst_HQ_3 predicts Instrumental as its primary stem.
        # audio-separator's normal file path derives Vocals as the residual;
        # reproduce that in memory so the live path can emit vocals directly.
        compensate = float(getattr(model, "compensate", 1.0))
        if primary.shape == mix.T.shape:
            primary = primary.T
        if primary.shape != mix.shape:
            raise RuntimeError(
                f"unexpected MDX demix shape: primary={primary.shape}, mix={mix.shape}"
            )
        return mix - primary * compensate

    def process(
        self,
        samples: np.ndarray,
        *,
        input_rate: int,
        output_samples: int,
    ) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples[:, None]
        if samples.ndim != 2 or samples.shape[0] == 0:
            raise ValueError(f"expected frame-major native PCM, got {samples.shape}")

        # MDX/RoFormer are trained on stereo 44.1 kHz. Preserve the native left
        # and right loopback channels instead of collapsing to 16 kHz mono before
        # separation. Mono devices are duplicated; uncommon >2ch loopback inputs
        # use the front L/R pair.
        if samples.shape[1] == 1:
            stereo = np.repeat(samples, 2, axis=1)
        else:
            stereo = samples[:, :2]
        mix = _resample(
            np.ascontiguousarray(stereo.T, dtype=np.float32),
            int(input_rate),
            MODEL_SAMPLE_RATE,
            axis=-1,
        )
        vocals = self._vocals_from_demix(self.model, mix)
        if vocals.ndim == 1:
            vocals_mono = vocals
        elif vocals.ndim == 2:
            if vocals.shape == mix.T.shape:
                vocals = vocals.T
            channel_axis = 0 if vocals.shape[0] <= 8 else 1
            vocals_mono = vocals.mean(axis=channel_axis)
        else:
            raise RuntimeError(f"unexpected separator output shape: {vocals.shape}")
        output = _resample(
            np.asarray(vocals_mono, dtype=np.float32),
            MODEL_SAMPLE_RATE,
            self.target_sample_rate,
            axis=-1,
        )
        return _fit_length(output, int(output_samples))

    def close(self):
        try:
            del self.model
            del self.separator
        except Exception:
            pass
        try:
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        except Exception:
            pass


def _make_backend(
    mode: str,
    model_dir: Path,
    target_sample_rate: int,
    window_seconds: float,
):
    if mode in MODEL_FILENAMES:
        return AudioSeparatorLiveBackend(
            mode,
            model_dir,
            target_sample_rate,
            window_seconds,
        )
    raise ValueError(f"Unsupported audio preprocessing mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--window-seconds", type=float, required=True)
    args = parser.parse_args()

    backend = _make_backend(
        args.mode,
        Path(args.model_dir),
        args.target_sample_rate,
        args.window_seconds,
    )
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
            request_meta = _read_exact(sys.stdin.buffer, 12)
            input_rate, channels, output_samples = struct.unpack("<III", request_meta)
            if input_rate <= 0 or channels <= 0 or count % channels:
                raise ValueError(
                    f"invalid preprocessing request: count={count}, "
                    f"rate={input_rate}, channels={channels}"
                )
            payload = _read_exact(sys.stdin.buffer, count * 4)
            samples = np.frombuffer(payload, dtype=np.float32).copy().reshape(
                count // channels, channels
            )
            try:
                output = backend.process(
                    samples,
                    input_rate=input_rate,
                    output_samples=output_samples,
                )
            except Exception:
                traceback.print_exc(file=sys.stderr)
                _PROTO_OUT.write(struct.pack("<I", 0))
                _PROTO_OUT.flush()
                return 2
            output = _fit_length(output, output_samples)
            _PROTO_OUT.write(struct.pack("<I", len(output)))
            _PROTO_OUT.write(output.astype(np.float32, copy=False).tobytes())
            _PROTO_OUT.flush()
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
