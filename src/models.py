from dataclasses import dataclass
from enum import Enum


class WhisperModel(str, Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"
    LARGE_V3_TURBO = "large-v3-turbo"  # Faster variant with near large-v3 quality


class OutputMode(str, Enum):
    PLAIN = "plain"            # Full transcript, no timestamps
    # Each segment prefixed with [HH:MM:SS.mmm --> HH:MM:SS.mmm]
    TIMESTAMPS = "timestamps"
    # Timestamps + speaker label (requires pyannote.audio)
    SPEAKERS = "speakers"


@dataclass
class TranscriptionConfig:
    file_path: str
    language: str
    model: WhisperModel = WhisperModel.LARGE_V3
    output_file: str = "transcript.txt"
    output_mode: OutputMode = OutputMode.PLAIN
    # Required only when output_mode=OutputMode.SPEAKERS
    hf_token: str | None = None
    # Hint for diarization; leave None to auto-detect
    num_speakers: int | None = None
    # "auto" detects CUDA, falls back to CPU
    device: str = "auto"
    compute_type: str = "auto"
    # 0 = auto (os.cpu_count() in sequential, divided evenly in parallel)
    cpu_threads: int = 0
    # 1 = sequential; N = split audio into N chunks processed in parallel
    parallel_chunks: int = 1
