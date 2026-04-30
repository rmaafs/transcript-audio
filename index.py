import os

from dotenv import load_dotenv

from src.models import OutputMode, TranscriptionConfig, WhisperModel
from src.transcriber import Transcriber

load_dotenv()

if __name__ == "__main__":
    config = TranscriptionConfig(
        file_path="video.mp4",          # Path to audio or video file
        language="es",                   # Language code: "es", "en", "fr", …
        model=WhisperModel.LARGE_V3,     # See WhisperModel enum for all options
        output_file="transcript.txt",    # Where to save the result
        # output_mode=OutputMode.PLAIN,    # PLAIN | TIMESTAMPS | SPEAKERS

        # Uncomment to enable speaker diarization (requires pyannote.audio):
        output_mode=OutputMode.SPEAKERS,
        hf_token=os.getenv("HF_TOKEN"),
        # num_speakers=2,                # Optional hint; auto-detected if omitted

        # M1 Max: 10 CPU cores (8P + 2E). 4 workers × 2 threads = 8 perf cores.
        # GPU (32-core Metal) is NOT supported by ctranslate2 — CPU only.
        parallel_chunks=4,
        cpu_threads=2,
    )

    Transcriber(config).run()
