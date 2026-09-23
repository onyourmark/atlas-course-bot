from knowledge import build_course_chunks, search_chunk_matches
from pilot_platform import extract_document_text

def test_windows_text_encodings_preserve_searchable_words():
    text = "Jev can be used in agentic development to choose which model to use."
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
        extracted = extract_document_text("lecture.txt", text.encode(encoding))
        assert extracted == text
        assert search_chunk_matches(
            "How can Jev be used in agentic development?",
            build_course_chunks("", {"lecture.txt": extracted}),
        )

def test_named_topic_survives_different_question_wording():
    chunks = build_course_chunks("", {"lecture.txt": "Jev chooses which model should handle a request."})
    assert search_chunk_matches("How can Jev be used in agentic development?", chunks)
    assert not search_chunk_matches("How can Saturn be used in agentic development?", chunks)

def test_multiple_relevant_passages_from_one_lecture():
    text = "Jev chooses a model. " * 100 + "Jev routes requests to the billing team. " * 100
    matches = search_chunk_matches("What is Jev?", build_course_chunks("", {"lecture.txt": text}))
    assert len(matches) > 1
    assert len({m["chunk_idx"] for m in matches}) == len(matches)

def test_existing_misdecoded_transcript_recovers_from_original(tmp_path):
    from pilot_platform import PilotStore, generate_encryption_key
    store = PilotStore(tmp_path, generate_encryption_key())
    store.initialize()
    _, token = store.create_invitation("test@northeastern.edu", "Test Professor")
    professor = store.accept_invitation(token, "a-secure-password")
    course = store.create_course(professor["id"], "Test", "TEST 1000", "Fall 2026")
    original = "Jev routes agentic requests to the appropriate model.".encode("utf-16")
    store.save_document(
        course_id=course["id"], owner_id=professor["id"], filename="lecture.txt",
        document_type="transcript", content=original,
        extracted_text=original.decode("utf-8", errors="replace"),
    )
    _, transcripts, _ = store.load_course_materials(course["id"])
    assert transcripts["lecture.txt"] == original.decode("utf-16")
    assert search_chunk_matches("How can Jev be used in agentic development?",
                                build_course_chunks("", transcripts))


def test_specific_topic_beats_repeated_general_course_terms():
    general = "Agentic AI uses an agentic AI loop with actions and observations. " * 150
    chunks = build_course_chunks("", {
        "chapter1.txt": general,
        "lecture.txt": "Jev selects a model or action from supplied choices.",
    })
    for question in (
        "How can Jev be used in Agentic AI?",
        "How can Jev be used in agentic AI?",
        "How can Jev be used in agentic development?",
    ):
        matches = search_chunk_matches(question, chunks, max_chunks=3)
        assert matches
        assert all("Jev" in match["text"] for match in matches)
    assert not search_chunk_matches("How can Saturn be used in Agentic AI?", chunks)


def test_tool_definition_beats_generic_agentic_context():
    chunks = build_course_chunks("", {
        "references.txt": "Agentic AI patterns include tool use. " * 150,
        "definition.txt": "A tool is a function a program can call to perform an action, such as looking up a policy.",
    })
    for question in ("What is a tool? In agentic AI", "What is a tool in Agentic AI?",
                     "What are tools in agentic AI?"):
        matches = search_chunk_matches(question, chunks)
        assert matches[0]["source"] == "definition.txt"
