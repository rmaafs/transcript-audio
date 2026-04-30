def run_diarization(
    file_path: str,
    hf_token: str,
    num_speakers: int | None = None,
):
    """
    Run speaker diarization using pyannote.audio.

    Requirements:
      pip install pyannote.audio
      Accept model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1
    """
    import os
    import tempfile
    from pathlib import Path

    try:
        import av as _av
        from pyannote.audio import Pipeline
    except ImportError:
        raise ImportError(
            "pyannote.audio is required for speaker diarization.\n"
            "Install it with: pip install pyannote.audio\n"
            "Then accept the model terms at:\n"
            "  https://huggingface.co/pyannote/speaker-diarization-3.1"
        )

    # Redirect HuggingFace cache to .tmp/huggingface/ inside the repo
    hf_cache = Path(__file__).resolve().parent.parent / ".tmp" / "huggingface"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )

    # Move pipeline to the best available device.
    # pyannote uses PyTorch (not ctranslate2), so MPS works on Apple Silicon.
    import torch
    if torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
        print("Diarization device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        print("Diarization device: CUDA")
    else:
        print("Diarization device: CPU")

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers

    # pyannote mis-counts samples on MP3/MP4; convert to PCM WAV first
    wav_path = _to_wav(file_path)
    try:
        return pipeline(wav_path, **kwargs)
    finally:
        Path(wav_path).unlink(missing_ok=True)


def _to_wav(file_path: str) -> str:
    """Convert any audio/video file to a 16 kHz mono PCM WAV temp file."""
    import array
    import tempfile
    import wave

    import av

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    with av.open(file_path) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

        with wave.open(tmp.name, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # s16 = 2 bytes
            wav.setframerate(16000)
            for packet in container.demux(stream):
                for frame in packet.decode():
                    for resampled in resampler.resample(frame):
                        wav.writeframes(resampled.to_ndarray().tobytes())

    return tmp.name


def build_speaker_map(diarization_result) -> dict[tuple[float, float], str]:
    """Build a dict of (turn_start, turn_end) -> speaker_label."""
    # pyannote >= 3.3 returns DiarizeOutput; extract the Annotation from it
    annotation = (
        diarization_result.speaker_diarization
        if hasattr(diarization_result, "speaker_diarization")
        else diarization_result
    )
    return {
        (turn.start, turn.end): speaker
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    }


def get_speaker_for_segment(
    segment_start: float,
    segment_end: float,
    speaker_map: dict[tuple[float, float], str],
) -> str:
    """Return the speaker label with the greatest overlap with the given segment."""
    best_speaker = "SPEAKER_00"
    best_overlap = 0.0

    for (turn_start, turn_end), speaker in speaker_map.items():
        overlap = max(0.0, min(segment_end, turn_end) -
                      max(segment_start, turn_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker

    return best_speaker
