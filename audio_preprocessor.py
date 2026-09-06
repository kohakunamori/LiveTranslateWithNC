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

# Keep windows aligned to the application's native 32 ms VAD chunks. Heavy
# source-separation/enhancement models need enough context to amortize per-call
# overhead, so they intentionally add a few seconds of latency.
_WINDOW_CHUNKS = {
    "demucs_v4": 250,  # 8.0 s; amortizes Demucs per-call overhead
    "clearvoice_mossformer2_se": 125,  # 4.0 s
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

    The public interface always consumes/produces the same mono float32 chunks as
    AudioCapture/VADProcessor. Heavy model dependencies live in isolated uv
    environments and are hosted in a persistent subprocess so model weights load
    once per mode switch instead of once per audio window.
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
        self.window_chunks = _WINDOW_CHUNKS.get(self.mode, 1)
        self.window_samples = self.chunk_samples * self.window_chunks
        self._pending = np.empty(0, dtype=np.float32)
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
        return self.window_samples / self.sample_rate

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
                    "--sample-rate",
                    str(self.sample_rate),
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

            # Heavy backends pay a large one-time CUDA/kernel warm-up cost. Do it
            # while the existing model-loading dialog is still visible instead of
            # stalling the first live capture window.
            if self.mode in ("demucs_v4", "clearvoice_mossformer2_se"):
                warmup_samples = min(self.window_samples, self.sample_rate * 4)
                t = np.arange(warmup_samples, dtype=np.float32) / self.sample_rate
                warmup = (1e-4 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
                log.info(f"Warming up audio preprocessor: {self.display_name}")
                self._request(warmup)

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

    def _request(self, samples: np.ndarray) -> np.ndarray:
        if self.mode == "off":
            return np.asarray(samples, dtype=np.float32).copy()
        self.start()
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise RuntimeError("audio preprocessor worker is unavailable")
        if proc.poll() is not None:
            raise RuntimeError(
                f"audio preprocessor worker exited unexpectedly: {proc.returncode}"
            )

        samples = np.ascontiguousarray(samples, dtype=np.float32)
        start = time.perf_counter()
        with self._io_lock:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"audio preprocessor worker exited unexpectedly: {proc.returncode}"
                )
            proc.stdin.write(struct.pack("<I", len(samples)))
            proc.stdin.write(samples.tobytes())
            proc.stdin.flush()
            header = _read_exact(proc.stdout, 4)
            count = struct.unpack("<I", header)[0]
            if count == 0:
                raise RuntimeError(f"{self.display_name} failed while processing audio")
            payload = _read_exact(proc.stdout, count * 4)
        output = np.frombuffer(payload, dtype=np.float32).copy()
        elapsed = time.perf_counter() - start
        duration = len(samples) / self.sample_rate
        rtf = elapsed / duration if duration else 0.0
        log.debug(
            f"Audio preprocess {self.mode}: {duration:.2f}s -> {elapsed:.2f}s "
            f"(RTF={rtf:.2f})"
        )
        if elapsed > duration:
            log.warning(
                f"Audio preprocessor slower than realtime: {self.display_name}, "
                f"RTF={rtf:.2f}"
            )
        if len(output) != len(samples):
            if len(output) > len(samples):
                output = output[: len(samples)]
            else:
                output = np.pad(output, (0, len(samples) - len(output)))
        return output.astype(np.float32, copy=False)

    def _processing_loop(self) -> None:
        while True:
            item = self._input_queue.get()
            try:
                if item is None:
                    return
                generation, block, valid_samples = item
                processed = self._request(block)
                self._output_queue.put((generation, processed, valid_samples))
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
        valid_samples: int,
        *,
        wait: bool = False,
    ) -> None:
        item = (
            self._generation,
            np.ascontiguousarray(block, dtype=np.float32),
            int(valid_samples),
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
                generation, processed, valid_samples = self._output_queue.get_nowait()
            except queue.Empty:
                break
            if generation != self._generation:
                continue
            output_chunks.extend(
                self._split_chunks(processed, valid_samples=valid_samples)
            )
        return output_chunks

    def _raise_worker_error(self) -> None:
        if self._worker_error is not None:
            exc = self._worker_error
            self._worker_error = None
            raise RuntimeError(
                f"{self.display_name} preprocessing worker failed: {exc}"
            ) from exc

    def _split_chunks(self, samples: np.ndarray, valid_samples: int | None = None):
        if valid_samples is not None:
            samples = samples[:valid_samples]
        chunks = []
        for start in range(0, len(samples), self.chunk_samples):
            chunk = samples[start : start + self.chunk_samples]
            if len(chunk) < self.chunk_samples:
                chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)))
            chunks.append(np.asarray(chunk, dtype=np.float32))
        return chunks

    def process_chunk(self, chunk: np.ndarray) -> list[np.ndarray]:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if self.mode == "off":
            return [chunk]
        self.start()
        self._raise_worker_error()
        if len(chunk) != self.chunk_samples:
            # AudioCapture normally produces exact chunks; normalize unusual tail
            # chunks so the worker/VAD boundary remains deterministic.
            if len(chunk) > self.chunk_samples:
                chunk = chunk[: self.chunk_samples]
            else:
                chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)))

        with self._lock:
            self._pending = np.concatenate((self._pending, chunk))
            while len(self._pending) >= self.window_samples:
                block = self._pending[: self.window_samples].copy()
                self._pending = self._pending[self.window_samples :]
                self._enqueue_block(block, len(block))
        return self._drain_output()

    def flush(self) -> list[np.ndarray]:
        if self.mode == "off":
            return []
        self.start()
        self._raise_worker_error()
        with self._lock:
            if len(self._pending):
                valid = len(self._pending)
                block = np.pad(self._pending, (0, self.window_samples - valid))
                self._pending = np.empty(0, dtype=np.float32)
                self._enqueue_block(block, valid, wait=True)
        self._input_queue.join()
        self._raise_worker_error()
        return self._drain_output()

    def reset(self) -> None:
        with self._lock:
            self._generation += 1
            self._pending = np.empty(0, dtype=np.float32)
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
            self._pending = np.empty(0, dtype=np.float32)
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
            # Demucs/ClearVoice request. Killing the isolated worker is safe.
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
