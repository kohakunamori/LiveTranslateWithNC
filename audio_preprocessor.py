from __future__ import annotations

import logging
import queue
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from model_manager import (
    audio_preprocessor_env_python,
    audio_preprocessor_model_dir,
    audio_preprocessor_display_name,
    is_audio_preprocessor_ready,
    normalize_audio_preprocess_mode,
)

log = logging.getLogger("LiveTranslate.AudioPreprocess")
APP_DIR = Path(__file__).parent

# Keep streaming windows aligned to the application's native 32 ms VAD chunks.
# python-audio-separator-live uses ~1.6 s windows and ~0.1 s overlap per edge;
# keep MDX just above the upstream ~1.5 s quality floor and RoFormer at 1.6 s.
_STREAM_CONFIG = {
    "mdx_net": {"window_chunks": 47, "overlap_chunks": 3},  # 1.504 s
    "melband_roformer": {"window_chunks": 50, "overlap_chunks": 3},  # 1.600 s
}


def _read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        data = stream.read(remaining)
        if not data:
            raise EOFError("audio preprocessor worker closed its pipe")
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)


class AudioPreprocessor:
    """Optional preprocessing stage inserted immediately before VAD.

    Off mode preserves the historical mono float32 capture contract.  Music
    modes consume native frame-major loopback PCM plus its sample rate, keep the
    microphone as a 16 kHz sidecar, and always emit the same mono 16 kHz chunks
    expected by VADProcessor. Heavy model dependencies live in an isolated uv
    environment and a persistent subprocess so weights load once per mode switch.
    """

    def __init__(
        self,
        mode: str = "off",
        *,
        sample_rate: int = 16000,
        chunk_duration: float = 0.032,
    ):
        self.mode = normalize_audio_preprocess_mode(mode)
        self.sample_rate = int(sample_rate)
        self.chunk_duration = float(chunk_duration)
        self.chunk_samples = max(1, round(self.sample_rate * self.chunk_duration))
        stream_config = _STREAM_CONFIG.get(
            self.mode, {"window_chunks": 1, "overlap_chunks": 0}
        )
        self.window_chunks = int(stream_config["window_chunks"])
        self.overlap_chunks = int(stream_config["overlap_chunks"])
        self.hop_chunks = max(1, self.window_chunks - 2 * self.overlap_chunks)
        self.window_samples = self.chunk_samples * self.window_chunks
        self.overlap_samples = self.chunk_samples * self.overlap_chunks
        self.hop_samples = self.chunk_samples * self.hop_chunks
        self._pending_native: list[np.ndarray] = []
        self._pending_mic: list[np.ndarray] = []
        self._input_rate: int | None = None
        self._input_channels: int | None = None
        self._has_scheduled_window = False
        self._proc: subprocess.Popen | None = None
        self._stderr_thread: threading.Thread | None = None
        self._processing_thread: threading.Thread | None = None
        self._input_queue: queue.Queue = queue.Queue(maxsize=3)
        self._output_queue: queue.Queue = queue.Queue()
        self._worker_error: Exception | None = None
        self._generation = 0
        self._lock = threading.RLock()
        self._io_lock = threading.Lock()

    @property
    def display_name(self) -> str:
        return audio_preprocessor_display_name(self.mode)

    @property
    def latency_seconds(self) -> float:
        if self.mode == "off":
            return 0.0
        return self.window_chunks * self.chunk_duration

    @property
    def started(self) -> bool:
        if self.mode == "off":
            return True
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        with self._lock:
            if self.mode == "off" or self.started:
                return
            if not is_audio_preprocessor_ready(self.mode):
                raise RuntimeError(f"Audio preprocessor is not ready: {self.display_name}")

            python = audio_preprocessor_env_python(self.mode)
            if python is None or not python.is_file():
                raise RuntimeError(f"Missing runtime for {self.display_name}")

            worker = APP_DIR / "audio_preprocess_worker.py"
            model_dir = audio_preprocessor_model_dir(self.mode)
            log.info(
                f"Loading audio preprocessor: {self.display_name} "
                f"(window={self.latency_seconds:.2f}s)"
            )
            self._proc = subprocess.Popen(
                [
                    str(python),
                    str(worker),
                    "--mode",
                    self.mode,
                    "--model-dir",
                    str(model_dir),
                    "--target-sample-rate",
                    str(self.sample_rate),
                    "--window-seconds",
                    str(self.latency_seconds),
                ],
                cwd=str(APP_DIR),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr, daemon=True
            )
            self._stderr_thread.start()
            assert self._proc.stdout is not None
            ready = _read_exact(self._proc.stdout, len(b"READY\n"))
            if ready != b"READY\n":
                code = self._proc.poll()
                self.close()
                raise RuntimeError(
                    f"Audio preprocessor failed to initialize: {self.display_name} "
                    f"(exit={code})"
                )

            # Pay CUDA/ORT kernel setup cost while the model-loading dialog is
            # visible, then report whether one inference fits inside the streaming
            # hop budget. This mirrors the upstream live separator's startup test.
            if self.mode in _STREAM_CONFIG:
                benchmark_rate = 44100
                benchmark_frames = max(
                    1, round(benchmark_rate * self.latency_seconds)
                )
                t = np.arange(benchmark_frames, dtype=np.float32) / benchmark_rate
                tone = (1e-4 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
                warmup = np.column_stack((tone, tone))
                log.info(f"Warming up audio preprocessor: {self.display_name}")
                self._request(
                    warmup,
                    input_rate=benchmark_rate,
                    output_samples=self.window_samples,
                    warn_slow=False,
                )
                benchmark_start = time.perf_counter()
                self._request(
                    warmup,
                    input_rate=benchmark_rate,
                    output_samples=self.window_samples,
                    warn_slow=False,
                )
                benchmark_elapsed = time.perf_counter() - benchmark_start
                hop_seconds = self.hop_chunks * self.chunk_duration
                rtf = benchmark_elapsed / max(hop_seconds, 1e-6)
                log.info(
                    f"Live separator benchmark: {self.display_name}, "
                    f"window={self.latency_seconds:.3f}s, "
                    f"hop={hop_seconds:.3f}s, inference={benchmark_elapsed:.3f}s, "
                    f"hop-RTF={rtf:.2f}"
                )
                if benchmark_elapsed >= hop_seconds:
                    log.warning(
                        f"{self.display_name} cannot keep up with realtime at the "
                        f"current window size (inference {benchmark_elapsed:.2f}s >= "
                        f"hop {hop_seconds:.2f}s)"
                    )

            self._worker_error = None
            self._processing_thread = threading.Thread(
                target=self._processing_loop,
                name=f"AudioPreprocess-{self.mode}",
                daemon=True,
            )
            self._processing_thread.start()
            log.info(f"Audio preprocessor ready: {self.display_name}")

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        while True:
            raw = proc.stderr.readline()
            if not raw:
                break
            text = raw.decode("utf-8", errors="replace").rstrip()
            if text:
                log.info(f"[{self.mode}] {text}")

    def _request(
        self,
        samples: np.ndarray,
        *,
        input_rate: int,
        output_samples: int,
        warn_slow: bool = True,
    ) -> np.ndarray:
        if self.mode == "off":
            mono = np.asarray(samples, dtype=np.float32)
            if mono.ndim > 1:
                mono = mono.mean(axis=1)
            return np.asarray(mono, dtype=np.float32).reshape(-1).copy()
        self.start()
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise RuntimeError("audio preprocessor worker is unavailable")
        if proc.poll() is not None:
            raise RuntimeError(
                f"audio preprocessor worker exited unexpectedly: {proc.returncode}"
            )
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples[:, None]
        if samples.ndim != 2:
            raise ValueError(f"native audio must be frame-major 2D PCM, got {samples.shape}")
        frames, channels = samples.shape
        if frames <= 0 or channels <= 0:
            return np.zeros(max(0, int(output_samples)), dtype=np.float32)
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        input_rate = int(input_rate)
        output_samples = max(1, int(output_samples))
        start = time.perf_counter()
        with self._io_lock:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"audio preprocessor worker exited unexpectedly: {proc.returncode}"
                )
            proc.stdin.write(
                struct.pack(
                    "<IIII",
                    int(frames * channels),
                    input_rate,
                    int(channels),
                    output_samples,
                )
            )
            proc.stdin.write(samples.tobytes())
            proc.stdin.flush()
            header = _read_exact(proc.stdout, 4)
            count = struct.unpack("<I", header)[0]
            if count == 0:
                raise RuntimeError(f"{self.display_name} failed while processing audio")
            payload = _read_exact(proc.stdout, count * 4)
        output = np.frombuffer(payload, dtype=np.float32).copy()
        elapsed = time.perf_counter() - start
        duration = frames / max(input_rate, 1)
        rtf = elapsed / duration if duration else 0.0
        log.debug(
            f"Audio preprocess {self.mode}: {duration:.2f}s -> {elapsed:.2f}s "
            f"(RTF={rtf:.2f})"
        )
        realtime_budget = (
            self.hop_chunks * self.chunk_duration
            if self.mode in _STREAM_CONFIG
            else duration
        )
        if warn_slow and elapsed > realtime_budget:
            log.warning(
                f"Audio preprocessor slower than realtime: {self.display_name}, "
                f"inference={elapsed:.2f}s > hop budget={realtime_budget:.2f}s"
            )
        if len(output) != output_samples:
            if len(output) > output_samples:
                output = output[:output_samples]
            else:
                output = np.pad(output, (0, output_samples - len(output)))
        return output.astype(np.float32, copy=False)

    def _processing_loop(self) -> None:
        while True:
            item = self._input_queue.get()
            try:
                if item is None:
                    return
                (
                    generation,
                    block,
                    input_rate,
                    mic_block,
                    output_start,
                    output_end,
                ) = item
                processed = self._request(
                    block,
                    input_rate=input_rate,
                    output_samples=len(mic_block),
                )
                selected = (
                    processed[int(output_start) : int(output_end)]
                    + mic_block[int(output_start) : int(output_end)]
                )
                self._output_queue.put((generation, selected))
            except Exception as exc:
                self._worker_error = exc
                log.error(
                    f"Audio preprocessing worker failed ({self.display_name}): {exc}",
                    exc_info=True,
                )
                # Any queued windows belong to a worker that can no longer be
                # trusted. Mark them done so flush()/shutdown cannot deadlock.
                while True:
                    try:
                        queued = self._input_queue.get_nowait()
                    except queue.Empty:
                        break
                    else:
                        self._input_queue.task_done()
                        if queued is None:
                            break
                return
            finally:
                self._input_queue.task_done()

    def _enqueue_block(
        self,
        block: np.ndarray,
        input_rate: int,
        mic_block: np.ndarray,
        output_start: int,
        output_end: int,
        *,
        wait: bool = False,
    ) -> None:
        item = (
            self._generation,
            np.ascontiguousarray(block, dtype=np.float32),
            int(input_rate),
            np.ascontiguousarray(mic_block, dtype=np.float32).reshape(-1),
            int(output_start),
            int(output_end),
        )
        if wait:
            self._input_queue.put(item)
            return
        try:
            self._input_queue.put_nowait(item)
        except queue.Full:
            # Staying live is more useful than allowing unbounded latency. Drop
            # the oldest window that has not started yet; the in-flight window is
            # never interrupted.
            try:
                dropped = self._input_queue.get_nowait()
            except queue.Empty:
                self._input_queue.put_nowait(item)
            else:
                self._input_queue.task_done()
                if dropped is None:
                    self._input_queue.put_nowait(dropped)
                    raise RuntimeError("audio preprocessor is shutting down")
                log.warning(
                    f"Audio preprocessing backlog full; dropped one "
                    f"{self.latency_seconds:.1f}s window to catch up"
                )
                self._input_queue.put_nowait(item)

    def _drain_output(self) -> list[np.ndarray]:
        output_chunks: list[np.ndarray] = []
        while True:
            try:
                generation, processed = self._output_queue.get_nowait()
            except queue.Empty:
                break
            if generation != self._generation:
                continue
            output_chunks.extend(self._split_chunks(processed))
        return output_chunks

    def _raise_worker_error(self) -> None:
        if self._worker_error is not None:
            exc = self._worker_error
            self._worker_error = None
            raise RuntimeError(
                f"{self.display_name} preprocessing worker failed: {exc}"
            ) from exc

    def _split_chunks(self, samples: np.ndarray):
        chunks = []
        for start in range(0, len(samples), self.chunk_samples):
            chunk = samples[start : start + self.chunk_samples]
            if len(chunk) < self.chunk_samples:
                chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)))
            chunks.append(np.asarray(chunk, dtype=np.float32))
        return chunks

    def _fit_mic_chunk(self, mic_chunk: np.ndarray | None) -> np.ndarray:
        if mic_chunk is None:
            return np.zeros(self.chunk_samples, dtype=np.float32)
        mic = np.asarray(mic_chunk, dtype=np.float32).reshape(-1)
        if len(mic) > self.chunk_samples:
            return mic[: self.chunk_samples].copy()
        if len(mic) < self.chunk_samples:
            return np.pad(mic, (0, self.chunk_samples - len(mic))).astype(np.float32)
        return mic.copy()

    def _clear_pending_locked(self) -> None:
        self._pending_native.clear()
        self._pending_mic.clear()
        self._input_rate = None
        self._input_channels = None
        self._has_scheduled_window = False

    def process_chunk(
        self,
        chunk: np.ndarray,
        *,
        input_rate: int | None = None,
        mic_chunk: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        chunk = np.asarray(chunk, dtype=np.float32)
        if self.mode == "off":
            return [chunk.reshape(-1)]
        self.start()
        self._raise_worker_error()
        if input_rate is None or int(input_rate) <= 0:
            raise ValueError("music preprocessing requires the native input sample rate")
        input_rate = int(input_rate)
        if chunk.ndim == 1:
            chunk = chunk[:, None]
        if chunk.ndim != 2 or chunk.shape[0] == 0 or chunk.shape[1] == 0:
            raise ValueError(f"music preprocessing requires frame-major native PCM, got {chunk.shape}")
        chunk = np.ascontiguousarray(chunk, dtype=np.float32)
        mic = self._fit_mic_chunk(mic_chunk)

        with self._lock:
            channels = int(chunk.shape[1])
            if self._input_rate is None:
                self._input_rate = input_rate
                self._input_channels = channels
            elif self._input_rate != input_rate or self._input_channels != channels:
                log.info(
                    "Native audio format changed; resetting separator overlap state: "
                    f"{self._input_rate}Hz/{self._input_channels}ch -> "
                    f"{input_rate}Hz/{channels}ch"
                )
                self._generation += 1
                self._clear_pending_locked()
                self._input_rate = input_rate
                self._input_channels = channels

            self._pending_native.append(chunk)
            self._pending_mic.append(mic)
            while len(self._pending_native) >= self.window_chunks:
                block = np.concatenate(
                    self._pending_native[: self.window_chunks], axis=0
                )
                mic_block = np.concatenate(
                    self._pending_mic[: self.window_chunks], axis=0
                )
                if self._has_scheduled_window:
                    output_start = self.overlap_samples
                else:
                    # Preserve the start of the stream; only the right edge needs
                    # to be held for the next overlapping window.
                    output_start = 0
                output_end = self.window_samples - self.overlap_samples
                del self._pending_native[: self.hop_chunks]
                del self._pending_mic[: self.hop_chunks]
                self._enqueue_block(
                    block,
                    input_rate,
                    mic_block,
                    output_start,
                    output_end,
                )
                self._has_scheduled_window = True
        return self._drain_output()

    def flush(self) -> list[np.ndarray]:
        if self.mode == "off":
            return []
        self.start()
        self._raise_worker_error()
        block = None
        mic_block = None
        input_rate = None
        output_start = 0
        output_end = 0
        with self._lock:
            if self._pending_native:
                valid_chunks = len(self._pending_native)
                input_rate = int(self._input_rate or 44100)
                channels = int(self._input_channels or 2)
                if self._has_scheduled_window:
                    output_start = self.overlap_samples
                else:
                    output_start = 0
                output_end = valid_chunks * self.chunk_samples
                if output_end > output_start:
                    native_chunks = list(self._pending_native)
                    mic_chunks = list(self._pending_mic)
                    missing = self.window_chunks - valid_chunks
                    native_frames = max(1, round(input_rate * self.chunk_duration))
                    for _ in range(max(0, missing)):
                        native_chunks.append(
                            np.zeros((native_frames, channels), dtype=np.float32)
                        )
                        mic_chunks.append(
                            np.zeros(self.chunk_samples, dtype=np.float32)
                        )
                    block = np.concatenate(native_chunks[: self.window_chunks], axis=0)
                    mic_block = np.concatenate(mic_chunks[: self.window_chunks], axis=0)
                self._clear_pending_locked()
        if block is not None:
            assert mic_block is not None and input_rate is not None
            self._enqueue_block(
                block,
                input_rate,
                mic_block,
                output_start,
                output_end,
                wait=True,
            )
        self._input_queue.join()
        self._raise_worker_error()
        return self._drain_output()

    def reset(self) -> None:
        with self._lock:
            self._generation += 1
            self._clear_pending_locked()
            while True:
                try:
                    item = self._input_queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._input_queue.task_done()
                    if item is None:
                        # Preserve shutdown sentinel if reset races with close().
                        try:
                            self._input_queue.put_nowait(None)
                        except queue.Full:
                            pass
                        break
            self._drain_output()

    def close(self) -> None:
        with self._lock:
            self._generation += 1
            self._clear_pending_locked()
            proc = self._proc
            if proc is None:
                return
            while True:
                try:
                    self._input_queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._input_queue.task_done()
            try:
                self._input_queue.put_nowait(None)
            except queue.Full:
                pass

        # If no inference owns the binary pipe, request a graceful worker exit.
        acquired = self._io_lock.acquire(timeout=0.2)
        if acquired:
            try:
                if proc.poll() is None and proc.stdin is not None:
                    proc.stdin.write(struct.pack("<I", 0))
                    proc.stdin.flush()
            except Exception:
                pass
            finally:
                self._io_lock.release()
        elif proc.poll() is None:
            # Mode switches must not block the Qt thread behind a multi-second
            # Live separator request. Killing the isolated worker is safe.
            proc.kill()

        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass

        thread = self._processing_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
        with self._lock:
            self._proc = None
            self._processing_thread = None
            self._worker_error = None
            self._drain_output()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
