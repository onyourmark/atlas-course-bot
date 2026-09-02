"""
Knowledge base loader for ATLAS.
Multi-course aware module for loading course materials, transcripts, and building chunks.
"""

import json
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
_TRANSCRIPT_EXTENSIONS = ["*.txt", "*.docx"]

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

    try:
        # Collect all supported files, deduplicating by stem
        seen_stems = set()
        for pattern in _TRANSCRIPT_EXTENSIONS:
            for file_path in sorted(transcripts_dir.glob(pattern)):
                if file_path.stem not in seen_stems:
                    seen_stems.add(file_path.stem)
                    if file_path.suffix.lower() == ".docx":
                        transcripts[file_path.name] = _read_docx(file_path)
                    else:
                        with open(file_path, "r") as f:
                            transcripts[file_path.name] = f.read()
    except Exception as e:
        print(f"Error loading transcripts for course {course_id}: {e}")

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


def search_chunk_matches(
    query: str,
    chunks: List[Dict],
    max_chunks: int = 4,
) -> List[Dict]:
    """Find relevant course sources and return one best excerpt per source."""
    if not chunks:
        return []

    terms = extract_search_terms(query)
    if not terms:
        return []

    scored = []
    for chunk in chunks:
        words = Counter(re.findall(r"\b[A-Za-z0-9][A-Za-z0-9_-]*\b", chunk["text"].lower()))
        matched_terms = [term for term in terms if words[term] > 0]
        required_term_count = 1 if len(terms) == 1 else 2
        if len(matched_terms) < required_term_count:
            continue

        distinct_match_score = len(matched_terms) * 10
        occurrence_score = sum(min(words[term], 5) for term in matched_terms)
        score = distinct_match_score + occurrence_score
        scored.append((score, chunk, matched_terms))

    scored.sort(key=lambda item: (-item[0], item[1]["display_name"], item[1]["chunk_idx"]))

    matches: List[Dict] = []
    used_sources = set()
    for score, chunk, matched_terms in scored:
        source_key = (chunk["source_type"], chunk["source"])
        if source_key in used_sources:
            continue
        used_sources.add(source_key)
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
