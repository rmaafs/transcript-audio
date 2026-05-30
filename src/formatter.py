import shutil
import subprocess
from pathlib import Path

from src.models import OutputMode

SAMPLE_MIN_SECONDS = 10.0
SAMPLE_MAX_SECONDS = 20.0


def format_segment(
    segment,
    mode: OutputMode,
    speaker: str | None = None,
) -> str:
    text = segment.text.strip()

    if mode == OutputMode.PLAIN:
        return text

    start = _format_time(segment.start)
    end = _format_time(segment.end)
    timestamp = f"[{start} --> {end}]"

    if mode == OutputMode.TIMESTAMPS:
        return f"{timestamp} {text}"

    # SPEAKERS mode
    label = speaker or "UNKNOWN"
    return f"{timestamp} {label}: {text}"


def save_transcript(lines: list[str], output_file: str, mode: OutputMode) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    separator = "\n\n" if mode == OutputMode.SPEAKERS else "\n"
    content_str = separator.join(line for line in lines if line)

    output_path.write_text(content_str, encoding="utf-8")
    print(f"\nTranscript saved to: {output_path.resolve()}")


def save_transcript_v2(
    segments_with_speakers: list[tuple[float, float, str, str]],
    output_file: str,
    source_audio_path: str | None = None,
) -> None:
    if not segments_with_speakers:
        return

    merged: list[tuple[float, float, str, str]] = []
    for start, end, speaker, text in segments_with_speakers:
        text = text.strip()
        if merged and merged[-1][2] == speaker:
            prev_start, _, prev_speaker, prev_text = merged[-1]
            merged[-1] = (prev_start, end, prev_speaker,
                          f"{prev_text} {text}".strip())
            continue
        merged.append((start, end, speaker, text))

    unique_speakers: list[str] = []
    for _, _, speaker, _ in merged:
        if speaker not in unique_speakers:
            unique_speakers.append(speaker)

    samples = _extract_speaker_samples(merged, unique_speakers, source_audio_path)

    print("\nAssign a display name for each speaker (press Enter to keep the label):")
    name_map: dict[str, str] = {}
    for speaker in unique_speakers:
        prompt_target = samples.get(speaker, speaker)
        answer = input(f"Quien es {prompt_target}?: ").strip()
        name_map[speaker] = answer or speaker

    lines = [
        f"[{_format_time(start)} --> {_format_time(end)}] {name_map[speaker]}: {text}"
        for start, end, speaker, text in merged
    ]

    base = Path(output_file)
    v2_path = base.with_name(f"{base.stem}_v2{base.suffix}")
    v2_path.parent.mkdir(parents=True, exist_ok=True)
    v2_path.write_text("\n\n".join(lines), encoding="utf-8")
    print(f"\nTranscript v2 saved to: {v2_path.resolve()}")


def _extract_speaker_samples(
    merged: list[tuple[float, float, str, str]],
    unique_speakers: list[str],
    source_audio_path: str | None,
) -> dict[str, str]:
    if not source_audio_path:
        return {}

    source = Path(source_audio_path)
    if not source.exists():
        return {}

    if shutil.which("ffmpeg") is None:
        print("\n[warning] ffmpeg not found in PATH; skipping speaker audio samples.")
        return {}

    out_dir = Path(".tmp") / "samples" / source.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    samples: dict[str, str] = {}
    for speaker in unique_speakers:
        fragment = _pick_sample_fragment(merged, speaker)
        if fragment is None:
            continue
        start, duration = fragment
        out_path = out_dir / f"{speaker}.mp3"
        if not _run_ffmpeg_extract(source, start, duration, out_path):
            continue
        samples[speaker] = str(out_path)

    return samples


def _pick_sample_fragment(
    merged: list[tuple[float, float, str, str]],
    speaker: str,
) -> tuple[float, float] | None:
    speaker_segments = [
        (start, end) for start, end, spk, _ in merged if spk == speaker
    ]
    if not speaker_segments:
        return None

    for start, end in speaker_segments:
        duration = end - start
        if duration >= SAMPLE_MIN_SECONDS:
            return start, min(duration, SAMPLE_MAX_SECONDS)

    start, end = max(speaker_segments, key=lambda s: s[1] - s[0])
    return start, min(end - start, SAMPLE_MAX_SECONDS)


def _run_ffmpeg_extract(
    source: Path, start: float, duration: float, out_path: Path
) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(source),
        "-vn",
        "-acodec", "libmp3lame",
        "-q:a", "4",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def _format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"
