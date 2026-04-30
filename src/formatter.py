from pathlib import Path

from src.models import OutputMode


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
    content = separator.join(line for line in lines if line)

    output_path.write_text(content, encoding="utf-8")
    print(f"\nTranscript saved to: {output_path.resolve()}")


def _format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"
