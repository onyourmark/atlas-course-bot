"""
ATLAS - Adaptive Teaching and Learning Assistant System
FastAPI backend for multi-course AI teaching assistants.

Built with FastAPI and the Anthropic and OpenAI APIs.
"""

import json
import hmac
import os
import re
import uuid
from dataclasses import dataclass
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
load_dotenv()

import anthropic
import openai
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from knowledge import (
    load_courses,
    load_syllabus,
    load_transcripts,
    load_concept_map,
    build_course_chunks,
    extract_search_terms,
    format_source_context,
    search_chunk_matches,
)
from prompts.system_prompt import build_system_prompt
from concept_maps import build_concept_map_prompt, parse_concept_map_response
from ai_providers import (
    DEFAULT_MODEL_BY_PROVIDER,
    PROVIDER_LABELS,
    model_catalog,
    model_name,
    normalize_provider,
    validate_provider_model,
)
from pilot_platform import (
    MAX_DOCUMENT_BYTES,
    PilotConfigurationError,
    PilotStore,
    PilotValidationError,
    build_store_from_environment,
    extract_document_text,
)


# -- Globals --

CLIENT: Optional[anthropic.Anthropic] = None
MODEL = os.getenv("ATLAS_MODEL", "claude-sonnet-4-6")
PILOT_ENABLED = os.getenv("ATLAS_PILOT_ENABLED", "false").lower() == "true"
PILOT_COOKIE_NAME = "atlas_pilot_session"
PILOT_STORE: Optional[PilotStore] = None
SECURE_COOKIES = os.getenv("ATLAS_SECURE_COOKIES", "true").lower() == "true"

# Multi-course data structures (keyed by course_id)
COURSES: Dict[str, Dict] = {}
SYSTEM_PROMPTS: Dict[str, str] = {}
CONCEPT_MAPS: Dict[str, Dict] = {}
COURSE_SOURCE_CHUNKS: Dict[str, List[Dict]] = {}

# Paths
BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
FEEDBACK_FILE = DATA_DIR / "feedback.json"
NO_MATERIALS_RESPONSE = (
    "The course materials I searched do not contain an answer to that question."
)


# -- Pydantic Models --

class ChatMessage(BaseModel):
    """A message in the conversation history."""
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=6000)


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    message: str = Field(min_length=1, max_length=4000)
    history: Optional[List[ChatMessage]] = Field(default=None, max_length=12)
    session_id: Optional[str] = Field(default=None, max_length=100)


class FeedbackRequest(BaseModel):
    """Request body for feedback endpoint."""
    course_id: str
    session_id: Optional[str] = Field(default=None, max_length=100)
    message: str = Field(max_length=8000)
    response: str = Field(max_length=20000)
    rating: Literal["up", "down"]
    comment: Optional[str] = Field(default=None, max_length=1000)


class FacultyLoginRequest(BaseModel):
    email: str = Field(max_length=320)
    password: str = Field(max_length=200)


class FacultyJoinRequest(BaseModel):
    token: str = Field(min_length=20, max_length=200)
    name: str = Field(default="", max_length=120)
    password: str = Field(max_length=200)


class FacultyApiKeyRequest(BaseModel):
    api_key: str = Field(max_length=500)


class FacultyCourseCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    code: str = Field(min_length=1, max_length=40)
    term: str = Field(min_length=1, max_length=80)
    section: str = Field(default="", max_length=80)
    campus: str = Field(default="Arlington", min_length=1, max_length=80)
    monthly_question_limit: int = Field(default=500, ge=1, le=5000)
    provider: Literal["anthropic", "openai"] = "anthropic"
    model: str = Field(default=DEFAULT_MODEL_BY_PROVIDER["anthropic"], max_length=100)


class FacultyCourseStatusRequest(BaseModel):
    status: Literal["draft", "published", "archived"]


class FacultyCourseModelRequest(BaseModel):
    provider: Literal["anthropic", "openai"]
    model: str = Field(min_length=1, max_length=100)


class PilotAdminLoginRequest(BaseModel):
    password: str = Field(max_length=200)


class PilotInvitationRequest(BaseModel):
    email: str = Field(max_length=320)
    name: str = Field(min_length=1, max_length=120)
    expires_hours: int = Field(default=72, ge=1, le=168)


class PilotTokenRequest(BaseModel):
    token: str = Field(min_length=20, max_length=200)


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

    # Reset the in-memory indexes before loading legacy and pilot courses.
    global COURSES, PILOT_STORE
    COURSES = load_courses()
    SYSTEM_PROMPTS.clear()
    CONCEPT_MAPS.clear()
    COURSE_SOURCE_CHUNKS.clear()
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

        # Build searchable chunks from the syllabus and lecture transcripts
        chunks = build_course_chunks(syllabus, transcripts)
        COURSE_SOURCE_CHUNKS[course_id] = chunks
        print(f"  - {len(chunks)} course source chunks built")

        # Build system prompt
        system_prompt = build_system_prompt(config, concept_map, syllabus)
        SYSTEM_PROMPTS[course_id] = system_prompt
        print(f"  - System prompt built ({len(system_prompt)} chars)")

        print()

    if PILOT_ENABLED:
        PILOT_STORE = build_store_from_environment()
        pilot_courses = PILOT_STORE.list_all_courses()
        for course in pilot_courses:
            _reload_pilot_course(course["id"])
        print(f"Loaded {len(pilot_courses)} faculty pilot courses\n")

    print("="*60)
    print("ATLAS ready to serve requests")
    print("="*60 + "\n")

    yield

    # Cleanup (if needed)
    print("ATLAS shutting down...")


app = FastAPI(lifespan=lifespan)

# -- CORS Middleware --

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ATLAS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type"],
    )


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault(
        "Permissions-Policy",
        "camera=(), microphone=(), geolocation=(), payment=()",
    )
    if request.url.scheme == "https":
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
        )
    if request.url.path.startswith(
        ("/faculty", "/pilot-admin", "/api/faculty", "/api/pilot-admin")
    ):
        response.headers["Cache-Control"] = "no-store"
    return response

# -- Helper Functions --

def _require_pilot_store() -> PilotStore:
    """Return the configured pilot store without exposing secret details."""
    if not PILOT_ENABLED or PILOT_STORE is None:
        raise HTTPException(status_code=404, detail="Faculty pilot not found")
    return PILOT_STORE


def _pilot_course_config(course: Dict) -> Dict:
    """Convert a persistent pilot record to the public course configuration."""
    store = _require_pilot_store()
    professor = store.get_professor(course["owner_id"])
    return {
        "name": course["name"],
        "code": course["code"],
        "professor": professor["name"] if professor else "Course instructor",
        "campus": course["campus"],
        "term": course["term"],
        "section": course["section"],
        "_managed": True,
        "_owner_id": course["owner_id"],
        "_status": course["status"],
        "_list_on_homepage": False,
        "_provider": course.get("provider") or "anthropic",
        "_model": course.get("model") or MODEL,
    }


def _reload_pilot_course(course_id: str) -> Dict:
    """Reload one persistent faculty course into the in-memory search index."""
    store = _require_pilot_store()
    course = store.get_course(course_id)
    if not course:
        raise PilotValidationError("Course not found.")

    syllabus, transcripts, concept_map = store.load_course_materials(course_id)
    config = _pilot_course_config(course)
    syllabus_documents = [
        document
        for document in store.list_documents(course_id)
        if document["document_type"] == "syllabus"
    ]
    syllabus_filename = (
        syllabus_documents[0]["filename"] if syllabus_documents else "syllabus"
    )
    chunks = build_course_chunks(
        syllabus,
        transcripts,
        syllabus_filename=syllabus_filename,
    )

    COURSES[course_id] = config
    CONCEPT_MAPS[course_id] = concept_map
    COURSE_SOURCE_CHUNKS[course_id] = chunks
    SYSTEM_PROMPTS[course_id] = build_system_prompt(config, concept_map, syllabus)
    return {
        "course_id": course_id,
        "source_chunks": len(chunks),
        "concept_count": len(
            {key: value for key, value in concept_map.items() if key != "_meta"}
        ),
    }


def _validate_course(course_id: str) -> Dict:
    """
    Validate that a course_id exists and return its config.
    Raises HTTPException if not found.
    """
    if course_id not in COURSES:
        raise HTTPException(status_code=404, detail=f"Course {course_id} not found")
    config = COURSES[course_id]
    if config.get("_managed") and config.get("_status") != "published":
        raise HTTPException(status_code=404, detail=f"Course {course_id} not found")
    return config


def _get_client(course_id: str) -> tuple[str, Any]:
    """Return the course's provider and a client using the correct saved key."""
    config = COURSES.get(course_id, {})
    if config.get("_managed"):
        store = _require_pilot_store()
        provider = config.get("_provider", "anthropic")
        api_key = store.decrypted_api_key(config["_owner_id"], provider)
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail="This course assistant is temporarily unavailable.",
            )
        if provider == "openai":
            return provider, openai.OpenAI(api_key=api_key)
        return provider, anthropic.Anthropic(api_key=api_key)

    global CLIENT
    if CLIENT is None:
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail="This course assistant is temporarily unavailable.",
            )
        CLIENT = anthropic.Anthropic(api_key=api_key)
    return "anthropic", CLIENT


@dataclass
class ProviderResponse:
    text: str
    input_tokens: int
    output_tokens: int


def _call_course_model(
    course_id: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    system_prompt: str = "",
) -> ProviderResponse:
    """Call Anthropic or OpenAI while returning one common response shape."""
    config = COURSES.get(course_id, {})
    model = config.get("_model", MODEL)
    provider, client = _get_client(course_id)

    if provider == "openai":
        request: Dict[str, Any] = {
            "model": model,
            "input": messages,
            "max_output_tokens": max_tokens,
            # ATLAS does not use provider-side conversation storage.
            "store": False,
        }
        if system_prompt:
            request["instructions"] = system_prompt
        response = client.responses.create(**request)
        return ProviderResponse(
            text=response.output_text,
            input_tokens=int(response.usage.input_tokens),
            output_tokens=int(response.usage.output_tokens),
        )

    request = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system_prompt:
        request["system"] = system_prompt
    response = client.messages.create(**request)
    text_blocks = [
        getattr(block, "text", "")
        for block in response.content
        if getattr(block, "text", "")
    ]
    return ProviderResponse(
        text="\n".join(text_blocks).strip(),
        input_tokens=int(response.usage.input_tokens),
        output_tokens=int(response.usage.output_tokens),
    )


def _session_token(request: Request) -> str:
    return request.cookies.get(PILOT_COOKIE_NAME, "")


def _require_session(request: Request, role: str) -> Dict:
    store = _require_pilot_store()
    session = store.session_details(_session_token(request))
    if not session or session["role"] != role:
        raise HTTPException(status_code=401, detail="Sign in required")
    return session


def _require_professor(request: Request) -> Dict:
    store = _require_pilot_store()
    session = _require_session(request, "professor")
    professor = store.get_professor(session.get("professor_id") or "")
    if not professor:
        raise HTTPException(status_code=401, detail="Sign in required")
    return professor


def _require_pilot_admin(request: Request) -> Dict:
    return _require_session(request, "admin")


def _set_session_cookie(response: JSONResponse, token: str) -> None:
    response.set_cookie(
        key=PILOT_COOKIE_NAME,
        value=token,
        max_age=12 * 60 * 60,
        httponly=True,
        secure=SECURE_COOKIES,
        samesite="strict",
        path="/",
    )


def _clear_session_cookie(response: JSONResponse) -> None:
    response.delete_cookie(
        key=PILOT_COOKIE_NAME,
        httponly=True,
        secure=SECURE_COOKIES,
        samesite="strict",
        path="/",
    )


def _public_course_config(course_id: str, config: Dict) -> Dict:
    return {
        "id": course_id,
        "name": config.get("name", ""),
        "code": config.get("code", ""),
        "professor": config.get("professor", ""),
        "campus": config.get("campus", ""),
        "term": config.get("term", ""),
        "section": config.get("section", ""),
    }


def _faculty_course_payload(store: PilotStore, course: Dict) -> Dict:
    documents = store.list_documents(course["id"], course["owner_id"])
    usage = store.monthly_usage(course["id"])
    return {
        "id": course["id"],
        "name": course["name"],
        "code": course["code"],
        "term": course["term"],
        "section": course["section"],
        "campus": course["campus"],
        "status": course["status"],
        "provider": course.get("provider") or "anthropic",
        "provider_name": PROVIDER_LABELS.get(
            course.get("provider") or "anthropic", course.get("provider", "")
        ),
        "model": course.get("model") or MODEL,
        "model_name": model_name(
            course.get("provider") or "anthropic", course.get("model") or MODEL
        ),
        "model_updated_at": course.get("model_updated_at"),
        "model_history": store.course_model_history(
            course["id"], course["owner_id"], limit=10
        ),
        "monthly_question_limit": course["monthly_question_limit"],
        "remaining_questions": store.remaining_questions(course["id"]),
        "usage": usage,
        "documents": documents,
        "concept_count": len(
            {
                key: value
                for key, value in CONCEPT_MAPS.get(course["id"], {}).items()
                if key != "_meta"
            }
        ),
        "student_path": (
            f"/course/{course['id']}" if course["status"] == "published" else None
        ),
        "created_at": course["created_at"],
        "updated_at": course["updated_at"],
    }


def _safe_session_id(candidate: Optional[str]) -> str:
    value = (candidate or "").strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{8,100}", value):
        return value
    return str(uuid.uuid4())


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


def _build_retrieval_query(message: str, history: Optional[List[ChatMessage]]) -> str:
    """Use a prior student question when the current message is a short follow-up."""
    current_terms = extract_search_terms(message)
    normalized_message = re.sub(
        r"\[[A-Z ]+MODE\]", " ", message, flags=re.IGNORECASE
    ).strip().lower().rstrip(".!?")
    non_content_messages = {
        "bye", "goodbye", "got it", "hello", "hey", "hi", "ok", "okay",
        "thank you", "thanks",
    }
    if not current_terms and normalized_message in non_content_messages:
        return message

    refers_back = bool(
        re.search(r"\b(this|that|it|its|these|those)\b", message, flags=re.IGNORECASE)
    )
    needs_prior_question = not current_terms or (refers_back and len(current_terms) <= 2)
    if not needs_prior_question or not history:
        return message

    prior_questions = [
        item.content for item in history
        if item.role == "user" and item.content.strip()
    ]
    if not prior_questions:
        return message
    return prior_questions[-1] + "\n" + message


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
    """Return courses intended for the public landing page."""
    courses_list = []
    for course_id, config in COURSES.items():
        if config.get("_managed") and not config.get("_list_on_homepage"):
            continue
        if config.get("_managed") and config.get("_status") != "published":
            continue
        courses_list.append(_public_course_config(course_id, config))
    return {"courses": courses_list}


@app.get("/course/{course_id}/metadata")
async def get_course_metadata(course_id: str):
    """Return safe metadata for one accessible course, including private-link courses."""
    config = _validate_course(course_id)
    return _public_course_config(course_id, config)


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
    config = _validate_course(course_id)
    session_id = _safe_session_id(request.session_id)

    system_prompt = SYSTEM_PROMPTS.get(course_id, "")
    chunks = COURSE_SOURCE_CHUNKS.get(course_id, [])

    if not system_prompt:
        raise HTTPException(
            status_code=500,
            detail=f"System prompt not initialized for course {course_id}"
        )

    # Search the syllabus and transcripts. Ordinary question words are ignored so
    # an unrelated source is not presented merely because it contains "what".
    retrieval_query = _build_retrieval_query(request.message, request.history)
    source_matches = search_chunk_matches(retrieval_query, chunks, max_chunks=3)
    source_payload = [
        {
            "name": match["display_name"],
            "type": match["source_type"],
            "excerpt": match["excerpt"],
        }
        for match in source_matches
    ]

    # If the student asked a substantive question and no course source matched,
    # answer deterministically instead of asking the language model to guess.
    if extract_search_terms(retrieval_query) and not source_matches:
        return JSONResponse({
            "session_id": session_id,
            "course_id": course_id,
            "response": NO_MATERIALS_RESPONSE,
            "sources": [],
            "materials_found": False,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
            },
        })

    # Build user message with retrieved context
    user_message = request.message
    if source_matches:
        retrieved_context = format_source_context(source_matches)
        user_message = (
            f"COURSE SOURCE EXCERPTS:\n{retrieved_context}\n\n"
            "Use only the course material above to answer the content of the "
            "student's question. These sources and short excerpts will be shown "
            "separately below your answer. If the excerpts do not support an "
            f"answer, say exactly: {NO_MATERIALS_RESPONSE}\n\n"
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

    if config.get("_managed"):
        store = _require_pilot_store()
        if store.remaining_questions(course_id) <= 0:
            raise HTTPException(
                status_code=429,
                detail="This course has reached its monthly ATLAS question limit.",
            )

    # Call the provider and model selected for this course.
    try:
        model = config.get("_model", MODEL)
        provider = config.get("_provider", "anthropic")
        response = _call_course_model(
            course_id=course_id,
            max_tokens=2048,
            system_prompt=system_prompt,
            messages=messages,
        )

        assistant_message = response.text

        if config.get("_managed"):
            _require_pilot_store().record_usage(
                course_id=course_id,
                professor_id=config["_owner_id"],
                session_id=session_id,
                provider=provider,
                model=model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )

        return JSONResponse({
            "session_id": session_id,
            "course_id": course_id,
            "response": assistant_message,
            "sources": source_payload,
            "materials_found": bool(source_payload),
            "usage": {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            }
        })

    except (anthropic.APIError, openai.APIError) as exc:
        print(
            f"AI provider error for course {course_id}: "
            f"{type(exc).__name__}"
        )
        raise HTTPException(
            status_code=502,
            detail="The course assistant could not complete that request. Please try again.",
        ) from exc


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
    config = _validate_course(request.course_id)
    if request.rating not in {"up", "down"}:
        raise HTTPException(status_code=400, detail="Invalid feedback rating")

    if config.get("_managed"):
        timestamp = datetime.now(timezone.utc).isoformat()
        _require_pilot_store().record_feedback(
            course_id=request.course_id,
            session_id=_safe_session_id(request.session_id),
            rating=request.rating,
            comment=request.comment or "",
        )
        return JSONResponse({
            "status": "feedback_recorded",
            "timestamp": timestamp,
        })

    feedback = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "course_id": request.course_id,
        "session_id": _safe_session_id(request.session_id),
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
async def get_admin_page(request: Request, key: Optional[str] = None):
    """Serve the legacy dashboard or redirect to the pilot administrator."""
    if PILOT_ENABLED:
        return RedirectResponse(url="/pilot-admin", status_code=303)
    _check_admin_access(request, key)

    admin_path = STATIC_DIR / "admin.html"
    if admin_path.exists():
        return FileResponse(admin_path)

    return JSONResponse({"message": "Admin panel"})


@app.get("/api/admin/stats")
async def get_admin_stats(request: Request, key: Optional[str] = None):
    """Return aggregate statistics from feedback.json."""
    _check_admin_access(request, key)

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


# -- Faculty Pilot --

@app.get("/faculty/login")
async def get_faculty_login_page():
    _require_pilot_store()
    path = STATIC_DIR / "faculty_login.html"
    return FileResponse(path) if path.exists() else JSONResponse(
        {"message": "Faculty login page not found"}, status_code=404
    )


@app.get("/faculty/join")
async def get_faculty_join_page():
    _require_pilot_store()
    path = STATIC_DIR / "faculty_join.html"
    return FileResponse(path) if path.exists() else JSONResponse(
        {"message": "Faculty join page not found"}, status_code=404
    )


@app.get("/faculty")
async def get_faculty_page(request: Request):
    _require_pilot_store()
    try:
        _require_professor(request)
    except HTTPException:
        return RedirectResponse(url="/faculty/login", status_code=303)
    path = STATIC_DIR / "faculty.html"
    return FileResponse(path) if path.exists() else JSONResponse(
        {"message": "Faculty dashboard not found"}, status_code=404
    )


@app.post("/api/faculty/login")
async def faculty_login(payload: FacultyLoginRequest):
    store = _require_pilot_store()
    professor = store.authenticate_professor(payload.email, payload.password)
    if not professor:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = store.create_session("professor", professor["id"])
    response = JSONResponse({"professor": professor})
    _set_session_cookie(response, token)
    return response


@app.post("/api/faculty/logout")
async def faculty_logout(request: Request):
    store = _require_pilot_store()
    store.delete_session(_session_token(request))
    response = JSONResponse({"status": "signed_out"})
    _clear_session_cookie(response)
    return response


@app.post("/api/faculty/invitation")
async def faculty_invitation(payload: PilotTokenRequest):
    invitation = _require_pilot_store().invitation_details(payload.token)
    if not invitation:
        raise HTTPException(status_code=404, detail="Invitation is invalid or expired")
    return invitation


@app.post("/api/faculty/join")
async def faculty_join(payload: FacultyJoinRequest):
    store = _require_pilot_store()
    professor = store.accept_invitation(
        payload.token,
        payload.password,
        payload.name,
    )
    token = store.create_session("professor", professor["id"])
    response = JSONResponse({"professor": professor}, status_code=201)
    _set_session_cookie(response, token)
    return response


@app.get("/api/faculty/me")
async def faculty_me(request: Request):
    return {
        "professor": _require_professor(request),
        "model_catalog": model_catalog(),
    }


def _set_faculty_api_key(
    professor: Dict, provider: str, api_key: str
) -> Dict:
    try:
        clean_provider = normalize_provider(provider)
    except ValueError as exc:
        raise PilotValidationError(str(exc)) from exc
    updated = _require_pilot_store().set_professor_api_key(
        professor["id"], api_key, clean_provider
    )
    return {"professor": updated}


def _delete_faculty_api_key(professor: Dict, provider: str) -> Dict:
    try:
        clean_provider = normalize_provider(provider)
    except ValueError as exc:
        raise PilotValidationError(str(exc)) from exc
    store = _require_pilot_store()
    published = [
        course
        for course in store.list_courses_for_professor(professor["id"])
        if course["status"] == "published" and course["provider"] == clean_provider
    ]
    if published:
        provider_name = "Anthropic" if clean_provider == "anthropic" else "OpenAI"
        raise HTTPException(
            status_code=409,
            detail=(
                f"Unpublish courses using {provider_name}, or switch them to another "
                "provider, before removing this key."
            ),
        )
    updated = store.delete_professor_api_key(professor["id"], clean_provider)
    return {"professor": updated}


@app.put("/api/faculty/api-keys/{provider}")
async def faculty_set_provider_api_key(
    provider: str, request: Request, payload: FacultyApiKeyRequest
):
    return _set_faculty_api_key(
        _require_professor(request), provider, payload.api_key
    )


@app.delete("/api/faculty/api-keys/{provider}")
async def faculty_delete_provider_api_key(provider: str, request: Request):
    return _delete_faculty_api_key(_require_professor(request), provider)


@app.put("/api/faculty/api-key")
async def faculty_set_api_key(request: Request, payload: FacultyApiKeyRequest):
    return _set_faculty_api_key(
        _require_professor(request), "anthropic", payload.api_key
    )


@app.delete("/api/faculty/api-key")
async def faculty_delete_api_key(request: Request):
    return _delete_faculty_api_key(_require_professor(request), "anthropic")


@app.get("/api/faculty/courses")
async def faculty_list_courses(request: Request):
    professor = _require_professor(request)
    store = _require_pilot_store()
    courses = [
        _faculty_course_payload(store, course)
        for course in store.list_courses_for_professor(professor["id"])
    ]
    return {"courses": courses}


@app.post("/api/faculty/courses")
async def faculty_create_course(
    request: Request,
    payload: FacultyCourseCreateRequest,
):
    professor = _require_professor(request)
    store = _require_pilot_store()
    try:
        provider, model = validate_provider_model(payload.provider, payload.model)
    except ValueError as exc:
        raise PilotValidationError(str(exc)) from exc
    if not professor["api_keys"][provider]["has_key"]:
        provider_name = "Anthropic" if provider == "anthropic" else "OpenAI"
        raise HTTPException(
            status_code=400,
            detail=f"Add your {provider_name} API key before creating this course.",
        )
    course = store.create_course(
        owner_id=professor["id"],
        name=payload.name,
        code=payload.code,
        term=payload.term,
        section=payload.section,
        campus=payload.campus,
        monthly_question_limit=payload.monthly_question_limit,
        provider=provider,
        model=model,
    )
    _reload_pilot_course(course["id"])
    return JSONResponse(
        _faculty_course_payload(store, course),
        status_code=201,
    )


@app.put("/api/faculty/courses/{course_id}/model")
async def faculty_set_course_model(
    course_id: str,
    request: Request,
    payload: FacultyCourseModelRequest,
):
    professor = _require_professor(request)
    store = _require_pilot_store()
    updated = store.set_course_model(
        course_id=course_id,
        owner_id=professor["id"],
        provider=payload.provider,
        model=payload.model,
    )
    _reload_pilot_course(course_id)
    return _faculty_course_payload(store, updated)


@app.post("/api/faculty/courses/{course_id}/documents")
async def faculty_upload_documents(
    course_id: str,
    request: Request,
    document_type: str = Form(...),
    files: List[UploadFile] = File(...),
):
    professor = _require_professor(request)
    store = _require_pilot_store()
    course = store.get_course(course_id)
    if not course or course["owner_id"] != professor["id"]:
        raise HTTPException(status_code=404, detail="Course not found")
    if document_type not in {"syllabus", "transcript"}:
        raise HTTPException(status_code=400, detail="Invalid document type")
    if not files or len(files) > 20:
        raise HTTPException(status_code=400, detail="Upload between 1 and 20 files")

    extracted_files = []
    for upload in files:
        content = await upload.read()
        filename = upload.filename or ""
        text = extract_document_text(filename, content)
        extracted_files.append((filename, content, text))

    saved = [
        store.save_document(
            course_id=course_id,
            owner_id=professor["id"],
            filename=filename,
            document_type=document_type,
            content=content,
            extracted_text=text,
        )
        for filename, content, text in extracted_files
    ]
    _reload_pilot_course(course_id)
    course = store.get_course(course_id)
    return {
        "documents": saved,
        "course": _faculty_course_payload(store, course),
    }


@app.delete("/api/faculty/courses/{course_id}/documents/{document_id}")
async def faculty_delete_document(
    course_id: str,
    document_id: str,
    request: Request,
):
    professor = _require_professor(request)
    store = _require_pilot_store()
    store.delete_document(document_id, course_id, professor["id"])
    course = store.get_course(course_id)
    if (
        course
        and course["status"] == "published"
        and not store.list_documents(course_id, professor["id"])
    ):
        store.set_course_status(course_id, professor["id"], "draft")
    _reload_pilot_course(course_id)
    return {"status": "deleted"}


@app.post("/api/faculty/courses/{course_id}/concept-map")
async def faculty_generate_concept_map(course_id: str, request: Request):
    professor = _require_professor(request)
    store = _require_pilot_store()
    course = store.get_course(course_id)
    if not course or course["owner_id"] != professor["id"]:
        raise HTTPException(status_code=404, detail="Course not found")
    provider = course.get("provider") or "anthropic"
    model = course.get("model") or MODEL
    if not professor["api_keys"][provider]["has_key"]:
        provider_name = "Anthropic" if provider == "anthropic" else "OpenAI"
        raise HTTPException(
            status_code=400,
            detail=f"Add your {provider_name} API key before generating a concept map.",
        )

    syllabus, transcripts, _ = store.load_course_materials(course_id)
    prompt = build_concept_map_prompt(course, syllabus, transcripts)
    try:
        response = _call_course_model(
            course_id=course_id,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        concept_map = parse_concept_map_response(
            response.text,
            course_id,
        )
        store.set_concept_map(course_id, professor["id"], concept_map)
        store.record_usage(
            course_id=course_id,
            professor_id=professor["id"],
            session_id="faculty-concept-map",
            provider=provider,
            model=model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            event_type="concept_map",
        )
        _reload_pilot_course(course_id)
        return {
            "status": "generated",
            "concept_count": len(concept_map) - 1,
        }
    except (anthropic.APIError, openai.APIError) as exc:
        print(
            f"AI provider concept-map error for course {course_id}: "
            f"{type(exc).__name__}"
        )
        raise HTTPException(
            status_code=502,
            detail="ATLAS could not generate the concept map. Please try again.",
        ) from exc


@app.post("/api/faculty/courses/{course_id}/status")
async def faculty_set_course_status(
    course_id: str,
    request: Request,
    payload: FacultyCourseStatusRequest,
):
    professor = _require_professor(request)
    store = _require_pilot_store()
    course = store.get_course(course_id)
    if not course or course["owner_id"] != professor["id"]:
        raise HTTPException(status_code=404, detail="Course not found")

    if payload.status == "published":
        provider = course.get("provider") or "anthropic"
        if not professor["api_keys"][provider]["has_key"]:
            provider_name = "Anthropic" if provider == "anthropic" else "OpenAI"
            raise HTTPException(
                status_code=400,
                detail=f"Add your {provider_name} API key before publishing.",
            )
        if not store.list_documents(course_id, professor["id"]):
            raise HTTPException(
                status_code=400,
                detail="Upload a syllabus or lecture transcript before publishing.",
            )

    updated = store.set_course_status(course_id, professor["id"], payload.status)
    _reload_pilot_course(course_id)
    return _faculty_course_payload(store, updated)


@app.get("/pilot-admin")
async def get_pilot_admin_page():
    _require_pilot_store()
    path = STATIC_DIR / "pilot_admin.html"
    return FileResponse(path) if path.exists() else JSONResponse(
        {"message": "Pilot admin page not found"}, status_code=404
    )


@app.post("/api/pilot-admin/login")
async def pilot_admin_login(payload: PilotAdminLoginRequest):
    store = _require_pilot_store()
    configured_password = os.getenv("ATLAS_ADMIN_PASSWORD", "")
    if not configured_password:
        raise HTTPException(status_code=503, detail="Pilot administrator is not configured")
    if not hmac.compare_digest(payload.password, configured_password):
        raise HTTPException(status_code=401, detail="Invalid password")
    token = store.create_session("admin")
    response = JSONResponse({"status": "signed_in"})
    _set_session_cookie(response, token)
    return response


@app.post("/api/pilot-admin/logout")
async def pilot_admin_logout(request: Request):
    store = _require_pilot_store()
    store.delete_session(_session_token(request))
    response = JSONResponse({"status": "signed_out"})
    _clear_session_cookie(response)
    return response


@app.get("/api/pilot-admin/summary")
async def pilot_admin_summary(request: Request):
    _require_pilot_admin(request)
    return _require_pilot_store().admin_summary()


@app.post("/api/pilot-admin/invitations")
async def pilot_admin_create_invitation(
    request: Request,
    payload: PilotInvitationRequest,
):
    _require_pilot_admin(request)
    invitation, token = _require_pilot_store().create_invitation(
        payload.email,
        payload.name,
        payload.expires_hours,
    )
    join_path = f"/faculty/join#token={token}"
    invitation["join_path"] = join_path
    public_base_url = os.getenv("ATLAS_PUBLIC_BASE_URL", "").strip().rstrip("/")
    base_url = public_base_url or str(request.base_url).rstrip("/")
    invitation["join_url"] = f"{base_url}{join_path}"
    return JSONResponse(invitation, status_code=201)


# -- Admin: Upload and Course Management --

def _check_admin_access(request: Request, key: Optional[str]) -> None:
    """Use a session in pilot mode and the configured key in legacy mode."""
    if PILOT_ENABLED:
        _require_pilot_admin(request)
        return
    configured_key = os.getenv("ATLAS_ADMIN_PASSWORD", "")
    if not configured_key:
        raise HTTPException(status_code=503, detail="Admin dashboard is not configured")
    if not key or not hmac.compare_digest(key, configured_key):
        raise HTTPException(status_code=403, detail="Unauthorized")


def _validate_legacy_course(course_id: str) -> Dict:
    config = COURSES.get(course_id)
    if not config:
        raise HTTPException(
            status_code=404,
            detail=f"Course {course_id} not found in courses.json",
        )
    if config.get("_managed"):
        raise HTTPException(
            status_code=400,
            detail="Use the faculty dashboard to manage pilot course materials.",
        )
    return config


def _reload_course(course_id: str) -> Dict:
    """
    Reload a single course's materials into the global dicts.
    Returns a summary of what was loaded.
    """
    config = _validate_legacy_course(course_id)

    syllabus = load_syllabus(course_id)
    transcripts = load_transcripts(course_id)
    concept_map = load_concept_map(course_id)

    CONCEPT_MAPS[course_id] = concept_map
    chunks = build_course_chunks(syllabus, transcripts)
    COURSE_SOURCE_CHUNKS[course_id] = chunks
    system_prompt = build_system_prompt(config, concept_map, syllabus)
    SYSTEM_PROMPTS[course_id] = system_prompt

    return {
        "course_id": course_id,
        "transcript_chunks": len(chunks),
        "concept_map_entries": len({k: v for k, v in concept_map.items() if k != "_meta"}),
        "system_prompt_chars": len(system_prompt),
    }


@app.get("/admin/upload")
async def get_upload_page(request: Request, key: Optional[str] = None):
    """Serve the upload/course management page (admin-protected)."""
    _check_admin_access(request, key)
    upload_path = STATIC_DIR / "upload.html"
    if upload_path.exists():
        return FileResponse(upload_path)
    return JSONResponse({"message": "Upload page not found"})


@app.get("/api/admin/courses")
async def admin_list_courses(request: Request, key: Optional[str] = None):
    """List all courses with their file inventory."""
    _check_admin_access(request, key)

    result = []
    for course_id, config in COURSES.items():
        if config.get("_managed"):
            continue
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

        chunk_count = len(COURSE_SOURCE_CHUNKS.get(course_id, []))

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
    request: Request,
    key: Optional[str] = None,
    course_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload a syllabus file (.md or .txt) for a course."""
    _check_admin_access(request, key)
    _validate_legacy_course(course_id)

    course_dir = KNOWLEDGE_DIR / course_id
    course_dir.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    if len(content) > MAX_DOCUMENT_BYTES:
        raise HTTPException(status_code=400, detail="The syllabus must be 25 MB or smaller")
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
    request: Request,
    key: Optional[str] = None,
    course_id: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """Upload one or more transcript files (.docx or .txt) for a course."""
    _check_admin_access(request, key)
    _validate_legacy_course(course_id)

    transcripts_dir = KNOWLEDGE_DIR / course_id / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Upload at most 20 transcripts")

    uploaded = []
    for file in files:
        filename = Path(file.filename or "").name
        ext = Path(filename).suffix.lower()
        if ext not in (".txt", ".docx"):
            continue

        content = await file.read()
        if len(content) > MAX_DOCUMENT_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"{filename} must be 25 MB or smaller",
            )
        dest = transcripts_dir / filename
        dest.write_bytes(content)
        uploaded.append({"name": filename, "size": len(content)})

    summary = _reload_course(course_id)

    return JSONResponse({
        "status": "uploaded",
        "files": uploaded,
        "course_id": course_id,
        "reload_summary": summary,
    })


@app.delete("/api/admin/file")
async def delete_file(
    request: Request,
    key: Optional[str] = None,
    course_id: str = "",
    filename: str = "",
    file_type: str = "",
):
    """Delete a transcript file from a course."""
    _check_admin_access(request, key)
    _validate_legacy_course(course_id)

    if file_type == "transcript":
        safe_name = Path(filename).name
        if not safe_name or safe_name != filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        target = KNOWLEDGE_DIR / course_id / "transcripts" / safe_name
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
async def reload_course(
    course_id: str,
    request: Request,
    key: Optional[str] = None,
):
    """Force-reload a course's materials without restarting the server."""
    _check_admin_access(request, key)
    _validate_legacy_course(course_id)
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


@app.exception_handler(PilotValidationError)
async def pilot_validation_error_handler(
    request: Request,
    exc: PilotValidationError,
):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(PilotConfigurationError)
async def pilot_configuration_error_handler(
    request: Request,
    exc: PilotConfigurationError,
):
    print(f"ATLAS pilot configuration error: {exc}")
    return JSONResponse(
        status_code=503,
        content={"detail": "The faculty pilot is temporarily unavailable."},
    )


# -- Entry Point --

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
