"""Conservative cleanup of caption exports without rewriting spoken content."""
import html
import re

_TIME = r"(?:\d{1,2}:)?\d{2}:\d{2}[.,]\d{3}"
_TIMING = re.compile(r"^\s*(" + _TIME + r")\s*-->\s*(" + _TIME + r")(?:\s+.*)?$")
_CUE_ID = re.compile(r"^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}(?:/[^\s]+)?$", re.I)
_VOICE = re.compile(r"<v(?:\.[^\s>]+)?\s+([^>]+)>", re.I)
_MARKUP = re.compile(r"</?(?:v|c|b|i|u|ruby|rt)(?:[.\s][^>]*)?>|<" + _TIME + r">", re.I)


def clean_caption_text(text: str) -> str:
    """Join caption fragments and retain a time reference per speaker/minute.

    Leave non-caption documents untouched. Original uploaded bytes are retained
    by the storage layer; this function only prepares the searchable text.
    """
    lines = text.lstrip("\ufeff").splitlines()
    if not any(_TIMING.match(line) for line in lines):
        return text
    if not (text.lstrip("\ufeff").startswith("WEBVTT") or _VOICE.search(text)
            or any(_CUE_ID.match(line.strip()) for line in lines)
            or re.search(r"(?m)^\s*\d+\s*\n\s*" + _TIME + r"\s*-->", text)):
        return text

    result, fragments = [], []
    group_speaker, group_start, group_seconds = None, None, None
    cue_start, cue_lines = None, []

    def seconds(stamp):
        parts = stamp.replace(",", ".").split(":")
        return sum(float(value) * (60 ** index) for index, value in enumerate(reversed(parts)))

    def flush_group():
        nonlocal fragments
        if fragments:
            label = f"[{group_start}]"
            if group_speaker:
                label += " " + group_speaker + ":"
            result.append(label + " " + " ".join(fragments))
            fragments = []

    def flush_cue():
        nonlocal cue_lines, group_speaker, group_start, group_seconds
        if not cue_lines:
            return
        raw = " ".join(line.strip() for line in cue_lines if line.strip())
        cue_lines = []
        names = list(dict.fromkeys(_VOICE.findall(raw)))
        speaker = " / ".join(html.unescape(name) for name in names)
        spoken = html.unescape(_MARKUP.sub("", raw)).strip()
        if not spoken:
            return
        when = seconds(cue_start)
        if fragments and (speaker != group_speaker or when - group_seconds >= 60):
            flush_group()
        if not fragments:
            group_speaker, group_start, group_seconds = speaker, cue_start, when
        fragments.append(spoken)

    skip_block = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            skip_block = False
            continue
        if stripped.startswith(("NOTE", "STYLE", "REGION")) and (
            stripped in {"NOTE", "STYLE", "REGION"} or stripped.startswith("NOTE ")
        ):
            skip_block = True
            continue
        if skip_block:
            continue
        timing = _TIMING.match(line)
        if timing:
            flush_cue()
            cue_start = timing.group(1).replace(",", ".")
            continue
        if _CUE_ID.match(stripped):
            continue
        if stripped.isdigit():
            following = next((item for item in lines[index + 1:] if item.strip()), "")
            if _TIMING.match(following):
                continue
        if stripped.startswith("WEBVTT"):
            continue
        if cue_start is None:
            result.append(line)
        else:
            cue_lines.append(line)
    flush_cue()
    flush_group()
    return "\n\n".join(result)
