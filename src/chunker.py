import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AudioChunk:
    path: str
    # actual start used for extraction (includes lead-in overlap)
    extract_start: float
    filter_start: float   # keep segments with abs_start >= filter_start
    # keep segments with abs_start < filter_end (inf for last chunk)
    filter_end: float


def split_audio(file_path: str, num_chunks: int, overlap: float = 2.0) -> list[AudioChunk]:
    """Split audio into num_chunks PCM WAV temp files with overlap."""
    import av

    with av.open(file_path) as container:
        total_duration = container.duration / 1_000_000.0

    chunk_duration = total_duration / num_chunks
    chunks = []

    for i in range(num_chunks):
        filter_start = i * chunk_duration
        filter_end = (i + 1) * chunk_duration if i < num_chunks - \
            1 else float("inf")
        extract_start = max(0.0, filter_start - overlap)
        extract_end = min(total_duration, (i + 1) * chunk_duration + overlap)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        _extract_wav(file_path, tmp.name, extract_start, extract_end)

        chunks.append(AudioChunk(
            path=tmp.name,
            extract_start=extract_start,
            filter_start=filter_start,
            filter_end=filter_end,
        ))

    return chunks


def cleanup_chunks(chunks: list[AudioChunk]) -> None:
    for chunk in chunks:
        Path(chunk.path).unlink(missing_ok=True)


def _extract_wav(input_path: str, output_path: str, start: float, end: float) -> None:
    import av

    with av.open(input_path) as container:
        audio_stream = next(s for s in container.streams if s.type == "audio")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        container.seek(int(start * 1_000_000), any_frame=True)

        with wave.open(output_path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)

            for packet in container.demux(audio_stream):
                if packet.pts is None:
                    continue
                pts_sec = float(packet.pts * audio_stream.time_base)
                if pts_sec > end:
                    break
                for frame in packet.decode():
                    for resampled in resampler.resample(frame):
                        wav.writeframes(resampled.to_ndarray().tobytes())


# ---------------------------------------------------------------------------
# Module-level worker — must be top-level for multiprocessing pickling
# ---------------------------------------------------------------------------

def transcribe_chunk_worker(args: tuple) -> list[dict]:
    """Transcribe one audio chunk. Returns segments with absolute timestamps."""
    (
        chunk_path, extract_start, filter_start, filter_end,
        language, model_name, device, compute_type, models_dir, cpu_threads,
        progress_queue, chunk_idx,
    ) = args

    from faster_whisper import WhisperModel as FW

    model = FW(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=models_dir,
        cpu_threads=cpu_threads,
    )
    segments_gen, _ = model.transcribe(
        chunk_path, language=language, vad_filter=True)

    results = []
    for s in segments_gen:
        abs_start = s.start + extract_start
        abs_end = s.end + extract_start
        if abs_start < filter_start:
            continue
        if abs_start >= filter_end:
            break
        results.append({"start": abs_start, "end": abs_end, "text": s.text})
        if progress_queue is not None:
            progress_queue.put((chunk_idx, abs_end))

    if progress_queue is not None:
        progress_queue.put((chunk_idx, None))  # sentinel: chunk done

    return results
