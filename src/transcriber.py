import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from src.diarization import build_speaker_map, get_speaker_for_segment, run_diarization
from src.formatter import format_segment, save_transcript, save_transcript_v2
from src.models import OutputMode, TranscriptionConfig


class Transcriber:
    def __init__(self, config: TranscriptionConfig) -> None:
        self.config = config
        self._validate()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> None:
        cached = self._load_whisper_cache()
        if cached is not None:
            print(f"Resuming from Whisper cache ({len(cached)} segments)")
            segments = cached
        elif self.config.parallel_chunks > 1:
            segments = self._transcribe_parallel()
            self._save_whisper_cache(segments)
        else:
            segments = self._transcribe_sequential()
            self._save_whisper_cache(segments)

        speaker_map: dict = {}
        if self.config.output_mode == OutputMode.SPEAKERS:
            print("\nRunning speaker diarization…")
            diarization = run_diarization(
                self.config.file_path,
                self.config.hf_token,  # type: ignore[arg-type]
                self.config.num_speakers,
            )
            speaker_map = build_speaker_map(diarization)

        lines = [
            format_segment(
                segment,
                self.config.output_mode,
                speaker=get_speaker_for_segment(
                    segment.start, segment.end, speaker_map)
                if self.config.output_mode == OutputMode.SPEAKERS
                else None,
            )
            for segment in segments
        ]

        save_transcript(lines, self.config.output_file,
                        self.config.output_mode)

        if self.config.output_mode == OutputMode.SPEAKERS:
            segments_with_speakers = [
                (
                    segment.start,
                    segment.end,
                    get_speaker_for_segment(
                        segment.start, segment.end, speaker_map) or "UNKNOWN",
                    segment.text,
                )
                for segment in segments
            ]
            save_transcript_v2(
                segments_with_speakers,
                self.config.output_file,
                source_audio_path=self.config.file_path,
            )

        self._clear_whisper_cache()

    # ------------------------------------------------------------------
    # Private — transcription
    # ------------------------------------------------------------------

    def _transcribe_sequential(self) -> list:
        cpu_threads = self._resolve_cpu_threads(num_workers=1)
        device, compute_type = self._resolve_device_compute()
        self._print_device_info(device, compute_type, cpu_threads)

        from faster_whisper import WhisperModel as FW

        models_dir = self._models_dir()
        print(f"Loading Whisper model: {self.config.model.value}")
        model = FW(
            self.config.model.value,
            device=device,
            compute_type=compute_type,
            download_root=str(models_dir),
            cpu_threads=cpu_threads,
        )

        print(f"Transcribing: {self.config.file_path}")
        segments_gen, info = model.transcribe(
            self.config.file_path,
            language=self.config.language,
            vad_filter=True,
        )
        print(
            f"Audio duration: {info.duration:.1f}s  |  "
            f"Detected language: {info.language} ({info.language_probability:.0%})"
        )
        return self._collect_with_progress(segments_gen, info.duration)

    def _transcribe_parallel(self) -> list:
        import queue
        import threading
        import multiprocessing
        import av
        from src.chunker import cleanup_chunks, split_audio, transcribe_chunk_worker

        n = self.config.parallel_chunks
        cpu_threads = self._resolve_cpu_threads(num_workers=n)
        device, compute_type = self._resolve_device_compute()
        self._print_device_info(device, compute_type, cpu_threads, workers=n)

        with av.open(self.config.file_path) as container:
            total_duration = container.duration / 1_000_000.0

        models_dir = str(self._models_dir())

        print(f"Splitting '{self.config.file_path}' into {n} chunks…")
        chunks = split_audio(self.config.file_path, n)
        filter_starts = [c.filter_start for c in chunks]

        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        stop_event = threading.Event()

        def _reader(pbar: tqdm) -> None:
            chunk_pos = list(filter_starts)
            active = n
            while active > 0 and not stop_event.is_set():
                try:
                    chunk_idx, abs_end = progress_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if abs_end is None:
                    active -= 1
                    continue
                new_pos = max(chunk_pos[chunk_idx], abs_end)
                delta = new_pos - chunk_pos[chunk_idx]
                chunk_pos[chunk_idx] = new_pos
                if delta > 0:
                    pbar.update(delta)

        args_list = [
            (
                chunk.path,
                chunk.extract_start,
                chunk.filter_start,
                chunk.filter_end,
                self.config.language,
                self.config.model.value,
                device,
                compute_type,
                models_dir,
                cpu_threads,
                progress_queue,
                i,
            )
            for i, chunk in enumerate(chunks)
        ]

        results: list[list[dict]] = [[] for _ in range(n)]
        try:
            with tqdm(
                total=round(total_duration, 1),
                desc="Transcribing",
                unit="s",
                bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}s [{elapsed}<{remaining!s}]",
                ncols=80,
            ) as pbar:
                reader_thread = threading.Thread(
                    target=_reader, args=(pbar,), daemon=True)
                reader_thread.start()

                with ProcessPoolExecutor(max_workers=n) as executor:
                    futures = {
                        executor.submit(transcribe_chunk_worker, args): i
                        for i, args in enumerate(args_list)
                    }
                    for future in as_completed(futures):
                        idx = futures[future]
                        results[idx] = future.result()

                stop_event.set()
                reader_thread.join(timeout=3)

                remaining = pbar.total - pbar.n  # type: ignore[operator]
                if remaining > 0:
                    pbar.update(remaining)
        finally:
            cleanup_chunks(chunks)
            manager.shutdown()

        merged = sorted(
            (item for chunk_result in results for item in chunk_result),
            key=lambda x: x["start"],
        )
        return [_SegmentProxy(r["start"], r["end"], r["text"]) for r in merged]

    def _collect_with_progress(self, segments_gen, duration: float) -> list:
        collected = []

        with tqdm(
            total=100,
            desc="Transcribing",
            unit="%",
            bar_format="{l_bar}{bar}| {n:.0f}/{total}% [{elapsed}<{remaining}]",
            ncols=80,
        ) as pbar:
            last_pct = 0
            for segment in segments_gen:
                collected.append(segment)
                current_pct = int(min(100, segment.end / duration * 100))
                delta = current_pct - last_pct
                if delta > 0:
                    pbar.update(delta)
                    last_pct = current_pct

            if last_pct < 100:
                pbar.update(100 - last_pct)

        return collected

    # ------------------------------------------------------------------
    # Private — helpers
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        if not Path(self.config.file_path).exists():
            raise FileNotFoundError(f"File not found: {self.config.file_path}")

        if self.config.output_mode == OutputMode.SPEAKERS and not self.config.hf_token:
            raise ValueError(
                "Speaker diarization requires a HuggingFace token.\n"
                "Set hf_token in TranscriptionConfig.\n"
                "Get a free token at: https://huggingface.co/settings/tokens"
            )

    def _resolve_device_compute(self) -> tuple[str, str]:
        device = self.config.device
        compute_type = self.config.compute_type

        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        return device, compute_type

    def _resolve_cpu_threads(self, num_workers: int = 1) -> int:
        threads = self.config.cpu_threads
        if threads == 0:
            threads = max(1, (os.cpu_count() or 1) // num_workers)
        return threads

    def _models_dir(self) -> Path:
        d = Path(__file__).resolve().parent.parent / ".tmp" / "models"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _print_device_info(
        self, device: str, compute_type: str, cpu_threads: int, workers: int = 1
    ) -> None:
        note = ""
        if device == "cpu":
            import platform
            if platform.processor() == "arm" or "Apple" in platform.version():
                note = " (Apple Silicon — ARM NEON, Metal/MPS not supported by ctranslate2)"

        parallel_info = f"  |  Workers: {workers}" if workers > 1 else ""
        print(
            f"Device: {device.upper()}  |  Compute type: {compute_type}"
            f"  |  CPU threads/worker: {cpu_threads}{parallel_info}{note}"
        )

    def _whisper_cache_path(self) -> Path:
        key = hashlib.md5(self.config.file_path.encode()).hexdigest()[:12]
        cache_dir = Path(__file__).resolve().parent.parent / ".tmp" / "whisper"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{key}.json"

    def _save_whisper_cache(self, segments: list) -> None:
        data = [{"start": s.start, "end": s.end, "text": s.text}
                for s in segments]
        self._whisper_cache_path().write_text(json.dumps(data), encoding="utf-8")

    def _load_whisper_cache(self) -> list | None:
        path = self._whisper_cache_path()
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [_SegmentProxy(r["start"], r["end"], r["text"]) for r in raw]

    def _clear_whisper_cache(self) -> None:
        self._whisper_cache_path().unlink(missing_ok=True)


class _SegmentProxy:
    """Lightweight stand-in for faster-whisper's NamedTuple segment."""

    __slots__ = ("start", "end", "text")

    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text
