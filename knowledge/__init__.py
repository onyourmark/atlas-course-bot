"""
Knowledge base loader for ATLAS.
Multi-course aware module for loading course materials, transcripts, and building chunks.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


# Constants for chunking
_CHUNK_SIZE = 1500
_CHUNK_OVERLAP = 300

# Base knowledge directory
KNOWLEDGE_DIR = Path(__file__).parent

# Supported transcript file extensions
_TRANSCRIPT_EXTENSIONS = [".txt", ".docx", ".md", ".pdf", ".pptx"]

# Ordinary question words should not make an unrelated lecture look relevant.
_SEARCH_STOP_WORDS = {
    "about", "also", "answer", "are", "can", "class", "claster", "could", "course",
    "did", "does", "example", "explain", "from", "give", "have", "help", "how",
    "hello", "into", "just", "know", "lecture", "materials", "mean", "more", "need",
    "okay", "please", "professor", "question", "really", "said", "say", "should",
    "show", "student", "tell", "thank", "thanks", "than", "that", "think",
    "the", "their", "them", "then", "there", "these", "they", "this", "those",
    "understand", "use", "used", "uses", "using", "want", "was", "way", "ways",
    "were", "what", "when", "where", "which", "who", "why", "will", "with",
    "work", "working", "works", "would", "yes", "you", "your",
}


def _read_docx(file_path: Path) -> str:
    """
    Read text content from a .docx file.

    Args:
        file_path: Path to the .docx file

    Returns:
        Extracted text content
    """
    try:
        from docx import Document
        doc = Document(str(file_path))
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def load_courses() -> Dict[str, Dict]:
    """
    Load the course registry from courses.json.

    Returns:
        Dictionary mapping course_id to course metadata
    """
    courses_file = KNOWLEDGE_DIR / "courses.json"
    if not courses_file.exists():
        return {}

    try:
        with open(courses_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading courses.json: {e}")
        return {}


def load_syllabus(course_id: str) -> str:
    """
    Load the syllabus for a specific course.

    Args:
        course_id: The course ID (e.g., "6105")

    Returns:
        Syllabus content as a string
    """
    syllabus_path = KNOWLEDGE_DIR / course_id / "syllabus.md"
    if not syllabus_path.exists():
        return ""

    try:
        with open(syllabus_path, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading syllabus for course {course_id}: {e}")
        return ""


def load_transcripts(course_id: str) -> Dict[str, str]:
    """
    Load all transcripts for a course (both .txt and .docx formats).

    Args:
        course_id: The course ID

    Returns:
        Dictionary mapping transcript filename to content
    """
    transcripts = {}
    transcripts_dir = KNOWLEDGE_DIR / course_id / "transcripts"

    if not transcripts_dir.exists():
        return transcripts

    from pilot_platform import extract_document_text

    # Preserve distinct source filenames, including files with the same stem.
    for file_path in sorted(transcripts_dir.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in _TRANSCRIPT_EXTENSIONS:
            try:
                transcripts[file_path.name] = extract_document_text(
                    file_path.name, file_path.read_bytes()
                )
            except Exception as exc:
                print(f"Error reading {file_path.name}: {exc}")

    return dict(sorted(transcripts.items()))


def load_concept_map(course_id: str) -> Dict:
    """
    Load the concept map for a course.

    Args:
        course_id: The course ID

    Returns:
        Concept map as a dictionary
    """
    concept_map_path = KNOWLEDGE_DIR / course_id / "concept_map.json"
    if not concept_map_path.exists():
        return {}

    try:
        with open(concept_map_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading concept map for course {course_id}: {e}")
        return {}


def _build_text_chunks(
    content: str,
    source: str,
    source_type: str,
    display_name: str,
) -> List[Dict]:
    """Split one course source into overlapping searchable chunks."""
    chunks = []
    content = " ".join(content.split())
    start = 0
    chunk_idx = 0

    while start < len(content):
        end = min(len(content), start + _CHUNK_SIZE)
        if end < len(content):
            word_boundary = content.rfind(" ", start + _CHUNK_SIZE // 2, end)
            if word_boundary > start:
                end = word_boundary
        chunk_text = content[start:end]
        if chunk_text.strip():
            chunks.append({
                "text": chunk_text,
                "source": source,
                "source_type": source_type,
                "display_name": display_name,
                "chunk_idx": chunk_idx,
            })
            chunk_idx += 1
        if end >= len(content):
            break
        next_start = max(start + 1, end - _CHUNK_OVERLAP)
        next_space = content.find(" ", next_start)
        start = next_space + 1 if next_space >= 0 else end

    return chunks


def build_transcript_chunks(transcripts: Dict[str, str]) -> List[Dict]:
    """
    Build overlapping chunks from transcripts.

    Args:
        transcripts: Dictionary mapping filename to transcript content

    Returns:
        List of chunk dictionaries with 'text', 'source', and 'chunk_idx' keys
    """
    chunks: List[Dict] = []
    for source, content in transcripts.items():
        chunks.extend(_build_text_chunks(
            content=content,
            source=source,
            source_type="transcript",
            display_name=f"Lecture transcript: {source}",
        ))

    return chunks


def build_course_chunks(
    syllabus: str,
    transcripts: Dict[str, str],
    syllabus_filename: str = "syllabus.md",
) -> List[Dict]:
    """Build searchable chunks from the syllabus and all lecture transcripts."""
    chunks: List[Dict] = []
    if syllabus.strip():
        chunks.extend(_build_text_chunks(
            content=syllabus,
            source=syllabus_filename,
            source_type="syllabus",
            display_name=f"Course syllabus ({syllabus_filename})",
        ))
    chunks.extend(build_transcript_chunks(transcripts))
    return chunks


def extract_search_terms(query: str) -> List[str]:
    """Return meaningful words used to search the course materials."""
    query = re.sub(r"\[[A-Z ]+MODE\]", " ", query, flags=re.IGNORECASE)
    terms: List[str] = []
    for original in re.findall(r"\b[A-Za-z0-9][A-Za-z0-9_-]*\b", query):
        term = original.lower()
        is_short_course_term = len(term) >= 2 and (
            any(char.isdigit() for char in term) or original.isupper()
        )
        if (len(term) >= 3 or is_short_course_term) and term not in _SEARCH_STOP_WORDS:
            if term not in terms:
                terms.append(term)
    return terms


def _make_excerpt(text: str, terms: List[str], max_chars: int = 320) -> str:
    """Create a short excerpt centered on the first matching search term."""
    if len(text) <= max_chars:
        return text.strip()

    lower_text = text.lower()
    positions = [lower_text.find(term) for term in terms if lower_text.find(term) >= 0]
    match_position = min(positions) if positions else 0
    start = max(0, match_position - max_chars // 3)
    end = min(len(text), start + max_chars)

    if start > 0:
        next_space = text.find(" ", start)
        if next_space >= 0 and next_space < end:
            start = next_space + 1
    if end < len(text):
        previous_space = text.rfind(" ", start, end)
        if previous_space > start:
            end = previous_space

    excerpt = text[start:end].strip()
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(text):
        excerpt += "..."
    return excerpt


def requested_lecture_number(query: str) -> Optional[int]:
    words = ("one two three four five six seven eight nine ten eleven twelve "
             "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty").split()
    match = re.search(r"\blecture\s+(\d+|" + "|".join(words) + r")\b",
                      query, re.IGNORECASE)
    if not match:
        return None
    value = match.group(1).lower()
    return int(value) if value.isdigit() else words.index(value) + 1


def _document_overview(query: str, chunks: List[Dict]) -> Optional[List[Dict]]:
    """Resolve an explicitly requested chapter/document before passage search."""
    if not re.search(r"\b(?:main points|key points|summary|summari[sz]e|overview|"
                     r"main ideas|key ideas|covers|covered|points)\b", query, re.IGNORECASE):
        return None
    sources = sorted({chunk["source"] for chunk in chunks})
    lower = query.lower()
    selected = [source for source in sources if source.lower() in lower]
    if requested_lecture_number(query) is not None:
        selected = sources
    chapter = re.search(r"\bchapter[ _-]*(\d+|one|two|three|four|five|six|seven|"
                        r"eight|nine|ten|eleven|twelve|thirteen|fourteen)\b",
                        lower)
    if not selected and chapter:
        words = ["one", "two", "three", "four", "five", "six", "seven",
                 "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen"]
        value = chapter.group(1)
        number = int(value) if value.isdigit() else words.index(value) + 1
        pattern = re.compile(r"chapter[ _-]*0*" + str(number) + r"(?!\d)", re.IGNORECASE)
        selected = [source for source in sources if pattern.search(source)]
        # The chapter text is preferable to a classroom slide adaptation.
        primary = [source for source in selected if Path(source).suffix.lower() == ".md"]
        if primary:
            selected = primary
    if not selected:
        # Do not replace a missing explicitly requested file with passing mentions.
        if chapter or re.search(r"\.(?:md|txt|docx|pptx|pdf)\b", lower):
            return []
        return None

    matches = []
    budget = 96000
    for source_index, source in enumerate(selected):
        allowance = budget // (len(selected) - source_index)
        passages = sorted((c for c in chunks if c["source"] == source),
                          key=lambda c: c["chunk_idx"])
        if not passages or budget <= 0:
            break
        text = passages[0]["text"]
        for passage in passages[1:]:
            following = passage["text"]
            overlap = 0
            for size in range(min(_CHUNK_OVERLAP, len(text), len(following)), 0, -1):
                if text.endswith(following[:size]):
                    overlap = size
                    break
            text += ("" if overlap else "\n") + following[overlap:]
        if len(text) > allowance:
            # Sample across the document instead of silently losing its ending.
            count = max(1, allowance // (_CHUNK_SIZE + 80))
            indexes = sorted({round(i * (len(passages) - 1) / max(1, count - 1))
                              for i in range(count)})
            text = "[Selected passages across the document; not the complete text.]\n"
            text += "\n\n".join(passages[i]["text"] for i in indexes)
            text = text[:allowance]
        else:
            text = "[Complete document text.]\n" + text
        budget -= len(text)
        matches.append({**passages[0], "text": text, "score": 100,
                        "excerpt": _make_excerpt(passages[0]["text"], [])})
    return matches


def search_chunk_matches(
    query: str,
    chunks: List[Dict],
    max_chunks: int = 4,
) -> List[Dict]:
    """Find relevant passages, allowing several passages from a long source."""
    if not chunks:
        return []

    lecture = requested_lecture_number(query)
    if lecture is not None:
        chunks = [c for c in chunks if lecture in c.get("lecture_numbers", [c.get("lecture_number")])]
        if re.search(r"\b(?:said|spoken|recording|transcript)\b", query, re.IGNORECASE):
            transcripts = [c for c in chunks if c.get("document_type") == "lecture_transcript"]
            if transcripts:
                chunks = transcripts
        if not chunks:
            return []

    overview = _document_overview(query, chunks)
    if overview is not None:
        return overview

    if lecture is not None:
        query = re.sub(r"\blecture\s+(?:\d+|[a-z]+)\b", "", query, flags=re.IGNORECASE)
    terms = extract_search_terms(query)
    if not terms:
        return []

    # Separate the question subject from an optional domain qualifier.
    # For example, "What is a tool? In agentic AI" asks about tools.
    focus = re.split(r"\b(?:in|within)\b", query, maxsplit=1, flags=re.IGNORECASE)[0]
    focus_terms = extract_search_terms(focus)
    if not focus_terms:
        focus_terms = terms

    # An explicitly named topic can match even when the surrounding wording
    # differs from the lecture. Never substitute an unrelated capitalized topic.
    named_terms = {
        word.lower() for word in re.findall(r"\b[A-Z][A-Za-z0-9_-]*\b", query)
        if word.lower() in terms
    }
    chunk_words = [
        Counter(re.findall(r"\b[A-Za-z0-9][A-Za-z0-9_-]*\b", chunk["text"].lower()))
        for chunk in chunks
    ]
    for words in chunk_words:
        for term in terms:
            if not term.endswith("s"):
                words[term] += words[term + "s"]
            elif len(term) > 3 and not term.endswith("ss"):
                words[term] += words[term[:-1]]
    frequencies = {
        term: sum(1 for words in chunk_words if words[term])
        for term in terms
    }
    # A rare named topic must not be displaced by common surrounding words.
    # Count across the whole course, rather than guessing from capitalization
    # alone which of several capitalized words is the specific subject.
    # Sentence-initial verbs such as "Investigate" are not named topics.
    first_word = re.search(r"\b[A-Za-z0-9][A-Za-z0-9_-]*\b", query)
    if first_word and not frequencies.get(first_word.group().lower(), 0):
        named_terms.discard(first_word.group().lower())
    anchors = set()
    if named_terms:
        rarest_frequency = min(frequencies[term] for term in named_terms)
        anchors = {term for term in named_terms
                   if frequencies[term] == rarest_frequency}

    if len(focus_terms) == 1:
        anchors = set(focus_terms)

    scored = []
    for chunk, words in zip(chunks, chunk_words):
        matched_terms = [term for term in terms if words[term] > 0]
        if anchors and not anchors.issubset(matched_terms):
            continue
        required_term_count = 1 if len(terms) == 1 else 2
        named_topic_match = bool(anchors) and anchors.issubset(matched_terms)
        if len(matched_terms) < required_term_count and not named_topic_match:
            continue

        score = sum(
            (math.log(1 + len(chunks) / (1 + frequencies[term])) + 1)
            * (1 + 0.1 * min(words[term], 5))
            for term in matched_terms
        )
        if len(focus_terms) == 1:
            subject_term = focus_terms[0]
            if subject_term.endswith("s") and not subject_term.endswith("ss"):
                subject_term = subject_term[:-1]
            subject = re.escape(subject_term)
            # Prefer explanations over bibliographies mentioning the same term.
            if re.search(r"\b" + subject + r"s?\s+(?:is|are|means|refers to)\b",
                         chunk["text"], flags=re.IGNORECASE):
                score += 10
        if re.search(r"\bhow\b.*\b(?:use|used)\b", query, re.IGNORECASE):
            practical_terms = ("routing", "route", "choose", "choices", "example",
                               "classify", "classification", "action")
            score += 2 * sum(bool(re.search(r"\b" + word + r"s?\b",
                                           chunk["text"], re.IGNORECASE))
                             for word in practical_terms)
        scored.append((score, chunk, matched_terms))

    scored.sort(key=lambda item: (-item[0], item[1]["display_name"], item[1]["chunk_idx"]))

    matches: List[Dict] = []
    used_passages = set()
    for score, chunk, matched_terms in scored:
        passage_key = (chunk["source_type"], chunk["source"], chunk["chunk_idx"])
        if passage_key in used_passages:
            continue
        used_passages.add(passage_key)
        matches.append({
            **chunk,
            "score": score,
            "excerpt": _make_excerpt(chunk["text"], matched_terms),
        })
        if len(matches) >= max_chunks:
            break

    return matches


def format_source_context(matches: List[Dict]) -> str:
    """Format selected course sources for the language model."""
    parts = []
    for index, match in enumerate(matches, start=1):
        parts.append(
            f"[SOURCE {index}: {match['display_name']}]\n{match['text']}"
        )
    return "\n\n---\n\n".join(parts)


def search_chunks(
    query: str,
    chunks: List[Dict],
    max_chunks: int = 10,
) -> str:
    """
    Simple keyword search for relevant chunks.
    Scores chunks by keyword overlap and returns top matches.

    Args:
        query: Search query string
        chunks: List of chunk dictionaries
        max_chunks: Maximum number of chunks to return

    Returns:
        Concatenated chunk content for relevant matches
    """
    return format_source_context(
        search_chunk_matches(query, chunks, max_chunks=max_chunks)
    )
