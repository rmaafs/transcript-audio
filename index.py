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
    )

    Transcriber(config).run()
