#!/usr/bin/env python3
"""
Build Course Concept Map Tool for ATLAS

Reads course materials (syllabus + transcripts) and uses Claude
to generate a structured concept map and save it to concept_map.json.

Usage:
    python tools/build_map.py --course 6105
    python tools/build_map.py --course 5002
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import anthropic


def load_course_materials(course_id: str, knowledge_dir: Path) -> str:
    """Load syllabus and transcripts for a course."""
    course_dir = knowledge_dir / course_id
    materials = []

    # Load syllabus
    syllabus_path = course_dir / "syllabus.md"
    if syllabus_path.exists():
        materials.append(f"=== SYLLABUS ===\n{syllabus_path.read_text()}")

    # Load transcripts
    transcripts_dir = course_dir / "transcripts"
    if transcripts_dir.exists():
        for transcript_file in sorted(transcripts_dir.glob("*")):
            if transcript_file.is_file() and transcript_file.name != ".gitkeep":
                try:
                    if transcript_file.suffix == ".docx":
                        from docx import Document
                        doc = Document(str(transcript_file))
                        text = "\n".join([p.text for p in doc.paragraphs])
                    else:
                        text = transcript_file.read_text()
                    materials.append(f"=== {transcript_file.name} ===\n{text}")
                except Exception as e:
                    print(f"Warning: Could not read {transcript_file.name}: {e}")

    return "\n\n".join(materials)


def build_concept_map(course_id: str, materials: str) -> dict:
    """
    Use Claude to analyze course materials and build a concept map.
    Returns a structured dictionary of concepts, relationships, and metadata.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = f"""
Analyze the following course materials and generate a comprehensive concept map.

The concept map should be a JSON object where:
- Keys are concept names (e.g., "Data Pipelines", "Feature Engineering")
- Values are objects with:
  - "description": Brief explanation of the concept
  - "lectures": List of lecture names where it appears
  - "prerequisites": List of prerequisite concepts that should be learned first
  - "related": List of related concepts

Additionally, include a "_meta" key with:
- "total_concepts": Total number of unique concepts
- "course_id": The course ID
- "generated_at": ISO timestamp

Course Materials for {course_id}:

{materials[:50000]}  (truncated if needed)

Return ONLY valid JSON, no other text.
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    # Parse the response
    response_text = response.content[0].text

    # Try to extract JSON if wrapped in markdown code blocks
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]

    try:
        concept_map = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"Error parsing Claude's JSON response: {e}")
        print(f"Response text: {response_text[:500]}")
        return {"error": "Failed to parse concept map"}

    # Add metadata
    from datetime import datetime
    concept_map["_meta"] = {
        "total_concepts": len([k for k in concept_map if k != "_meta"]),
        "course_id": course_id,
        "generated_at": datetime.utcnow().isoformat(),
    }

    return concept_map


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build concept map for a course"
    )
    parser.add_argument(
        "--course",
        type=str,
        default="6105",
        help="Course ID (default: 6105)",
    )

    args = parser.parse_args()
    course_id = args.course

    # Paths
    base_dir = Path(__file__).parent.parent
    knowledge_dir = base_dir / "knowledge"

    # Validate course directory
    course_dir = knowledge_dir / course_id
    if not course_dir.exists():
        print(f"Error: Course directory not found: {course_dir}")
        return 1

    print(f"Building concept map for course {course_id}...")

    # Load materials
    materials = load_course_materials(course_id, knowledge_dir)
    if not materials.strip():
        print(f"Warning: No course materials found for {course_id}")
        print("Skipping concept map generation.")
        return 0

    print(f"Loaded {len(materials)} characters of course materials")

    # Build concept map using Claude
    print("Calling Claude API to generate concept map...")
    concept_map = build_concept_map(course_id, materials)

    if "error" in concept_map:
        print(f"Error: {concept_map['error']}")
        return 1

    # Save to file
    output_path = course_dir / "concept_map.json"
    with open(output_path, "w") as f:
        json.dump(concept_map, f, indent=2)

    print(f"Concept map saved to {output_path}")
    print(f"Total concepts: {concept_map['_meta']['total_concepts']}")

    return 0


if __name__ == "__main__":
    exit(main())
