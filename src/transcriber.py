import hashlib
import json
from pathlib import Path

from tqdm import tqdm

from src.diarization import build_speaker_map, get_speaker_for_segment, run_diarization
from src.formatter import format_segment, save_transcript
from src.models import OutputMode, TranscriptionConfig


class Transcriber:
    def __init__(self, config: TranscriptionConfig) -> None:
        self.config = config
        self._validate()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> None:
        print(f"Loading Whisper model: {self.config.model.value}")
        model = self._load_model()

        print(f"Transcribing: {self.config.file_path}")

        cached = self._load_whisper_cache()
        if cached is not None:
            print(f"Resuming from Whisper cache ({len(cached)} segments)")
            segments = cached
        else:
            segments_gen, info = model.transcribe(
                self.config.file_path,
                language=self.config.language,
                vad_filter=True,
            )
            print(
                f"Audio duration: {info.duration:.1f}s  |  "
                f"Detected language: {info.language} ({info.language_probability:.0%})"
            )
            segments = self._collect_with_progress(segments_gen, info.duration)
            self._save_whisper_cache(segments)
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
        self._clear_whisper_cache()

    # ------------------------------------------------------------------
    # Private
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

    def _load_model(self):
        from faster_whisper import WhisperModel as FasterWhisperModel

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

        device_note = ""
        if device == "cpu":
            import platform
            if platform.processor() == "arm" or "Apple" in platform.version():
                device_note = " (Apple Silicon — ARM NEON, Metal/MPS not supported by ctranslate2)"

        print(
            f"Device: {device.upper()}  |  Compute type: {compute_type}{device_note}")

        models_dir = Path(__file__).resolve().parent.parent / ".tmp" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        return FasterWhisperModel(
            self.config.model.value,
            device=device,
            compute_type=compute_type,
            download_root=str(models_dir),
        )

    def _whisper_cache_path(self) -> Path:
        key = hashlib.md5(self.config.file_path.encode()).hexdigest()[:12]
        cache_dir = Path(__file__).resolve().parent.parent / ".tmp" / "whisper"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{key}.json"

    def _save_whisper_cache(self, segments: list) -> None:
        data = [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in segments
        ]
        self._whisper_cache_path().write_text(json.dumps(data), encoding="utf-8")

    def _load_whisper_cache(self) -> list | None:
        path = self._whisper_cache_path()
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [_SegmentProxy(r["start"], r["end"], r["text"]) for r in raw]

    def _clear_whisper_cache(self) -> None:
        self._whisper_cache_path().unlink(missing_ok=True)

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


class _SegmentProxy:
    """Lightweight stand-in for faster-whisper's NamedTuple segment."""

    __slots__ = ("start", "end", "text")

    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text
