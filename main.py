"""
ATLAS - Adaptive Teaching and Learning Assistant System
FastAPI backend for multi-course AI teaching assistants.

Built with FastAPI, Claude API, and Anthropic SDK.
"""

import json
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

import anthropic
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from knowledge import (
    load_courses,
    load_syllabus,
    load_transcripts,
    load_concept_map,
    build_transcript_chunks,
    search_chunks,
)
from prompts.system_prompt import build_system_prompt


# -- Globals --

CLIENT = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"

# Multi-course data structures (keyed by course_id)
COURSES: Dict[str, Dict] = {}
SYSTEM_PROMPTS: Dict[str, str] = {}
CONCEPT_MAPS: Dict[str, Dict] = {}
TRANSCRIPT_CHUNKS: Dict[str, List[Dict]] = {}

# Paths
BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
FEEDBACK_FILE = DATA_DIR / "feedback.json"


# -- Pydantic Models --

class ChatMessage(BaseModel):
    """A message in the conversation history."""
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    message: str
    history: Optional[List[ChatMessage]] = None
    session_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    """Request body for feedback endpoint."""
    course_id: str
    session_id: Optional[str] = None
    message: str
    response: str
    rating: str  # "up" or "down"
    comment: Optional[str] = None


# -- Initialization --

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all courses and their materials on startup.
    Print initialization stats.
    """
    print("\n" + "="*60)
    print("ATLAS - Adaptive Teaching and Learning Assistant System")
    print("="*60 + "\n")

    # Load courses registry
    global COURSES
    COURSES = load_courses()
    print(f"Loaded {len(COURSES)} courses from courses.json\n")

    # For each course, load materials and build structures
    for course_id, config in COURSES.items():
        print(f"Initializing {course_id} ({config.get('code', 'UNKNOWN')})...")

        # Load materials
        syllabus = load_syllabus(course_id)
        transcripts = load_transcripts(course_id)
        concept_map = load_concept_map(course_id)

        # Build concept map section (just for system prompt)
        CONCEPT_MAPS[course_id] = concept_map

        # Build transcript chunks
        chunks = build_transcript_chunks(transcripts)
        TRANSCRIPT_CHUNKS[course_id] = chunks
        print(f"  - {len(chunks)} transcript chunks built")

        # Build system prompt
        system_prompt = build_system_prompt(config, concept_map, syllabus)
        SYSTEM_PROMPTS[course_id] = system_prompt
        print(f"  - System prompt built ({len(system_prompt)} chars)")

        print()

    print("="*60)
    print("ATLAS ready to serve requests")
    print("="*60 + "\n")

    yield

    # Cleanup (if needed)
    print("ATLAS shutting down...")


app = FastAPI(lifespan=lifespan)

# -- CORS Middleware --

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Helper Functions --

def _validate_course(course_id: str) -> Dict:
    """
    Validate that a course_id exists and return its config.
    Raises HTTPException if not found.
    """
    if course_id not in COURSES:
        raise HTTPException(status_code=404, detail=f"Course {course_id} not found")
    return COURSES[course_id]


def _append_feedback(feedback: Dict) -> None:
    """
    Thread-safe append of feedback to feedback.json.
    Creates file if it doesn't exist.
    """
    try:
        # Read existing feedback
        feedback_list = []
        if FEEDBACK_FILE.exists():
            with open(FEEDBACK_FILE, "r") as f:
                feedback_list = json.load(f)

        # Append new feedback
        feedback_list.append(feedback)

        # Write back
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(feedback_list, f, indent=2)
    except Exception as e:
        print(f"Error appending feedback: {e}")


def _read_feedback() -> List[Dict]:
    """Read all feedback from feedback.json."""
    if not FEEDBACK_FILE.exists():
        return []
    try:
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading feedback: {e}")
        return []


# -- Routes --

@app.get("/")
async def get_landing_page():
    """Serve the landing page with course list."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({
        "message": "Welcome to ATLAS",
        "courses": list(COURSES.keys())
    })


@app.get("/courses")
async def get_courses():
    """Return list of all courses with metadata."""
    courses_list = []
    for course_id, config in COURSES.items():
        courses_list.append({
            "id": course_id,
            **config
        })
    return {"courses": courses_list}


@app.get("/course/{course_id}")
async def get_course_page(course_id: str):
    """Serve the chat UI for a specific course."""
    _validate_course(course_id)
    course_path = STATIC_DIR / "course.html"
    if course_path.exists():
        return FileResponse(course_path)
    return JSONResponse({
        "message": f"Chat interface for {course_id}",
        "course_id": course_id
    })


@app.post("/course/{course_id}/chat")
async def chat(course_id: str, request: ChatRequest):
    """
    Main chat endpoint. Processes a user message and returns an AI response.
    """
    course_config = _validate_course(course_id)
    session_id = request.session_id or str(uuid.uuid4())

    system_prompt = SYSTEM_PROMPTS.get(course_id, "")
    chunks = TRANSCRIPT_CHUNKS.get(course_id, [])

    if not system_prompt:
        raise HTTPException(
            status_code=500,
            detail=f"System prompt not initialized for course {course_id}"
        )

    # Search for relevant transcript chunks
    retrieved_context = ""
    if chunks:
        retrieved_context = search_chunks(request.message, chunks, max_chunks=10)

    # Build user message with retrieved context
    user_message = request.message
    if retrieved_context:
        user_message = (
            f"RELEVANT LECTURE EXCERPTS:\n{retrieved_context}\n\n"
            f"STUDENT QUESTION:\n{request.message}"
        )

    # Build conversation history
    messages = []
    if request.history:
        for msg in request.history:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })

    messages.append({
        "role": "user",
        "content": user_message
    })

    # Call Claude API
    try:
        response = CLIENT.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=system_prompt,
            messages=messages,
        )

        assistant_message = response.content[0].text
        usage = response.usage

        return JSONResponse({
            "session_id": session_id,
            "course_id": course_id,
            "response": assistant_message,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            }
        })

    except anthropic.APIError as e:
        raise HTTPException(status_code=500, detail=f"API error: {str(e)}")


@app.get("/course/{course_id}/concept-map")
async def get_concept_map(course_id: str):
    """Return the concept map for a course as an array of concepts."""
    _validate_course(course_id)
    raw_map = CONCEPT_MAPS.get(course_id, {})
    concepts = []
    for name, info in raw_map.items():
        if name == "_meta":
            continue
        concepts.append({
            "id": name.lower().replace(" ", "_"),
            "name": name,
            "description": info.get("description", ""),
            "lectures": info.get("lectures", []),
            "prerequisites": info.get("prerequisites", []),
        })
    return {"course_id": course_id, "concepts": concepts}


@app.post("/feedback")
async def post_feedback(request: FeedbackRequest):
    """Log user feedback on a response."""
    _validate_course(request.course_id)

    feedback = {
        "timestamp": datetime.utcnow().isoformat(),
        "course_id": request.course_id,
        "session_id": request.session_id or "unknown",
        "message": request.message,
        "response": request.response,
        "rating": request.rating,
        "comment": request.comment or "",
    }

    _append_feedback(feedback)

    return JSONResponse({
        "status": "feedback_recorded",
        "timestamp": feedback["timestamp"]
    })


@app.get("/admin")
async def get_admin_page(key: Optional[str] = None):
    """Serve admin dashboard (protected with simple key check)."""
    if key != "atlas2026":
        raise HTTPException(status_code=403, detail="Unauthorized")

    admin_path = STATIC_DIR / "admin.html"
    if admin_path.exists():
        return FileResponse(admin_path)

    return JSONResponse({"message": "Admin panel"})


@app.get("/api/admin/stats")
async def get_admin_stats(key: Optional[str] = None):
    """Return aggregate statistics from feedback.json."""
    if key != "atlas2026":
        raise HTTPException(status_code=403, detail="Unauthorized")

    feedback_list = _read_feedback()

    unique_sessions = set(f.get("session_id") for f in feedback_list)
    approval_count = sum(1 for f in feedback_list if f.get("rating") == "up")

    # Per-course breakdown as array (matching frontend expectations)
    per_course = {}
    per_course_sessions = {}
    for feedback in feedback_list:
        cid = feedback.get("course_id", "unknown")
        if cid not in per_course:
            per_course[cid] = {"feedback_count": 0, "approval_count": 0}
            per_course_sessions[cid] = set()
        per_course[cid]["feedback_count"] += 1
        if feedback.get("rating") == "up":
            per_course[cid]["approval_count"] += 1
        sid = feedback.get("session_id")
        if sid:
            per_course_sessions[cid].add(sid)

    courses_array = []
    for cid, cstats in per_course.items():
        config = COURSES.get(cid, {})
        courses_array.append({
            "code": config.get("code", cid),
            "name": config.get("name", "Unknown"),
            "feedback_count": cstats["feedback_count"],
            "approval_count": cstats["approval_count"],
            "session_count": len(per_course_sessions.get(cid, set())),
        })

    # Recent feedback (last 20)
    recent = sorted(feedback_list, key=lambda x: x.get("timestamp", ""), reverse=True)[:20]
    recent_feedback = []
    for fb in recent:
        recent_feedback.append({
            "timestamp": fb.get("timestamp", ""),
            "course_id": fb.get("course_id", "unknown"),
            "rating": 1 if fb.get("rating") == "up" else -1,
            "user_message": fb.get("message", ""),
            "comment": fb.get("comment", ""),
        })

    return JSONResponse({
        "total_sessions": len(unique_sessions),
        "total_feedback": len(feedback_list),
        "approval_count": approval_count,
        "active_courses": len(COURSES),
        "courses": courses_array,
        "recent_feedback": recent_feedback,
    })

@app.get("/about")
async def get_about_page():
    """Serve the about page."""
    about_path = STATIC_DIR / "about.html"
    if about_path.exists():
        return FileResponse(about_path)

    return JSONResponse({
        "title": "About ATLAS",
        "description": "Adaptive Teaching and Learning Assistant System"
    })


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "courses_loaded": len(COURSES),
    })


# -- Admin: Upload and Course Management --

def _check_admin_key(key: Optional[str]) -> None:
    """Validate admin key or raise 403."""
    if key != "atlas2026":
        raise HTTPException(status_code=403, detail="Unauthorized")


def _reload_course(course_id: str) -> Dict:
    """
    Reload a single course's materials into the global dicts.
    Returns a summary of what was loaded.
    """
    config = COURSES.get(course_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Course {course_id} not found in courses.json")

    syllabus = load_syllabus(course_id)
    transcripts = load_transcripts(course_id)
    concept_map = load_concept_map(course_id)

    CONCEPT_MAPS[course_id] = concept_map
    chunks = build_transcript_chunks(transcripts)
    TRANSCRIPT_CHUNKS[course_id] = chunks
    system_prompt = build_system_prompt(config, concept_map, syllabus)
    SYSTEM_PROMPTS[course_id] = system_prompt

    return {
        "course_id": course_id,
        "transcript_chunks": len(chunks),
        "concept_map_entries": len({k: v for k, v in concept_map.items() if k != "_meta"}),
        "system_prompt_chars": len(system_prompt),
    }


@app.get("/admin/upload")
async def get_upload_page(key: Optional[str] = None):
    """Serve the upload/course management page (admin-protected)."""
    _check_admin_key(key)
    upload_path = STATIC_DIR / "upload.html"
    if upload_path.exists():
        return FileResponse(upload_path)
    return JSONResponse({"message": "Upload page not found"})


@app.get("/api/admin/courses")
async def admin_list_courses(key: Optional[str] = None):
    """List all courses with their file inventory."""
    _check_admin_key(key)

    result = []
    for course_id, config in COURSES.items():
        course_dir = KNOWLEDGE_DIR / course_id

        syllabus_path = course_dir / "syllabus.md"
        syllabus_exists = syllabus_path.exists() and syllabus_path.stat().st_size > 0

        transcripts_dir = course_dir / "transcripts"
        transcript_files = []
        if transcripts_dir.exists():
            for f in sorted(transcripts_dir.iterdir()):
                if f.suffix.lower() in (".txt", ".docx") and f.name != ".gitkeep":
                    transcript_files.append({
                        "name": f.name,
                        "size": f.stat().st_size,
                        "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                    })

        concept_map_path = course_dir / "concept_map.json"
        concept_count = 0
        if concept_map_path.exists():
            try:
                with open(concept_map_path) as f:
                    cmap = json.load(f)
                    concept_count = len({k: v for k, v in cmap.items() if k != "_meta"})
            except Exception:
                pass

        chunk_count = len(TRANSCRIPT_CHUNKS.get(course_id, []))

        result.append({
            "id": course_id,
            **config,
            "has_syllabus": syllabus_exists,
            "transcript_files": transcript_files,
            "transcript_count": len(transcript_files),
            "concept_count": concept_count,
            "chunk_count": chunk_count,
        })

    return {"courses": result}


@app.post("/api/admin/upload/syllabus")
async def upload_syllabus(
    key: Optional[str] = None,
    course_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload a syllabus file (.md or .txt) for a course."""
    _check_admin_key(key)
    _validate_course(course_id)

    course_dir = KNOWLEDGE_DIR / course_id
    course_dir.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    syllabus_path = course_dir / "syllabus.md"
    syllabus_path.write_bytes(content)

    summary = _reload_course(course_id)

    return JSONResponse({
        "status": "uploaded",
        "file": "syllabus.md",
        "course_id": course_id,
        "size": len(content),
        "reload_summary": summary,
    })


@app.post("/api/admin/upload/transcripts")
async def upload_transcripts(
    key: Optional[str] = None,
    course_id: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """Upload one or more transcript files (.docx or .txt) for a course."""
    _check_admin_key(key)
    _validate_course(course_id)

    transcripts_dir = KNOWLEDGE_DIR / course_id / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    uploaded = []
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in (".txt", ".docx"):
            continue

        content = await file.read()
        dest = transcripts_dir / file.filename
        dest.write_bytes(content)
        uploaded.append({"name": file.filename, "size": len(content)})

    summary = _reload_course(course_id)

    return JSONResponse({
        "status": "uploaded",
        "files": uploaded,
        "course_id": course_id,
        "reload_summary": summary,
    })


@app.delete("/api/admin/file")
async def delete_file(
    key: Optional[str] = None,
    course_id: str = "",
    filename: str = "",
    file_type: str = "",
):
    """Delete a transcript file from a course."""
    _check_admin_key(key)
    _validate_course(course_id)

    if file_type == "transcript":
        target = KNOWLEDGE_DIR / course_id / "transcripts" / filename
    elif file_type == "syllabus":
        target = KNOWLEDGE_DIR / course_id / "syllabus.md"
    else:
        raise HTTPException(status_code=400, detail="Invalid file_type")

    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")

    target.unlink()

    summary = _reload_course(course_id)

    return JSONResponse({
        "status": "deleted",
        "file": filename or "syllabus.md",
        "course_id": course_id,
        "reload_summary": summary,
    })


@app.post("/api/admin/reload/{course_id}")
async def reload_course(course_id: str, key: Optional[str] = None):
    """Force-reload a course's materials without restarting the server."""
    _check_admin_key(key)
    _validate_course(course_id)
    summary = _reload_course(course_id)
    return JSONResponse({"status": "reloaded", **summary})


# -- Static Files --

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# -- Error Handlers --

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


# -- Entry Point --

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
