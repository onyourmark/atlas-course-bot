"""
Knowledge base loader for ATLAS.
Multi-course aware module for loading course materials, transcripts, and building chunks.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional


# Constants for chunking
_CHUNK_SIZE = 1500
_CHUNK_OVERLAP = 300

# Base knowledge directory
KNOWLEDGE_DIR = Path(__file__).parent

# Supported transcript file extensions
_TRANSCRIPT_EXTENSIONS = ["*.txt", "*.docx"]


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
                        transcripts[file_path.stem] = _read_docx(file_path)
                    else:
                        with open(file_path, "r") as f:
                            transcripts[file_path.stem] = f.read()
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


def build_transcript_chunks(transcripts: Dict[str, str]) -> List[Dict]:
    """
    Build overlapping chunks from transcripts.

    Args:
        transcripts: Dictionary mapping filename to transcript content

    Returns:
        List of chunk dictionaries with 'text', 'source', and 'chunk_idx' keys
    """
    chunks = []

    for source, content in transcripts.items():
        # Remove extra whitespace
        content = " ".join(content.split())

        # Create overlapping chunks
        start = 0
        chunk_idx = 0
        while start < len(content):
            end = start + _CHUNK_SIZE
            chunk_text = content[start:end]
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "chunk_idx": chunk_idx,
                })
                chunk_idx += 1
            start += _CHUNK_SIZE - _CHUNK_OVERLAP

    return chunks


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
    if not chunks:
        return ""

    # Extract meaningful keywords (3+ chars)
    keywords = set(
        word.lower()
        for word in re.findall(r'\b\w+\b', query)
        if len(word) >= 3
    )

    if not keywords:
        return ""

    # Score chunks by keyword overlap
    scored = []
    for chunk in chunks:
        chunk_lower = chunk["text"].lower()
        score = sum(chunk_lower.count(kw) for kw in keywords)
        if score > 0:
            scored.append((score, chunk))

    if not scored:
        return ""

    # Sort by score descending and take top max_chunks
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for _, chunk in scored[:max_chunks]]

    # Format into a context string
    parts = []
    for chunk in top_chunks:
        source_label = chunk["source"].replace("_", " ").title()
        parts.append(f"[From {source_label}]\n{chunk['text']}")

    return "\n\n---\n\n".join(parts)
