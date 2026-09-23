from unittest.mock import patch
from fastapi.testclient import TestClient
import pytest
import main
from pilot_platform import PilotStore, generate_encryption_key, PilotValidationError
from knowledge import build_course_chunks, search_chunk_matches

@pytest.fixture
def setup(tmp_path):
    store = PilotStore(tmp_path, generate_encryption_key())
    store.initialize()
    _, token = store.create_invitation("test@northeastern.edu", "Instructor")
    professor = store.accept_invitation(token, "a-secure-password")
    course = store.create_course(professor["id"], "Test Course", "TEST1000", "Fall 2026")
    return store, professor, course

def test_upload_label_relabel_and_owner_checks(setup):
    store, professor, course = setup
    with patch.object(main, "_require_pilot_store", return_value=store), \
         patch.object(main, "_require_professor", return_value=professor), \
         patch.object(main, "COURSES", {}), patch.object(main, "COURSE_SOURCE_CHUNKS", {}), \
         patch.object(main, "CONCEPT_MAPS", {}), patch.object(main, "SYSTEM_PROMPTS", {}):
        client = TestClient(main.app)
        url = "/api/faculty/courses/" + course["id"] + "/documents"
        assert client.post(url, data={"document_type":"lecture_transcript"},
                           files={"files":("part.txt",b"Routing choices")}).status_code == 400
        response = client.post(url, data={"document_type":"lecture_transcript","lecture_number":"3"},
                     files=[("files",("part-a.txt",b"Routing chooses a department.")),
                            ("files",("part-b.txt",b"Validation checks the decision."))])
        assert response.status_code == 200, response.text
        documents = response.json()["documents"]
        chunks = main.COURSE_SOURCE_CHUNKS[course["id"]]
        for query in ("What are the main points from lecture three?",
                      "List twenty points from lecture 3."):
            matches = search_chunk_matches(query, chunks)
            assert {m["source"] for m in matches} == {"part-a.txt", "part-b.txt"}
        assert not search_chunk_matches("Main points from lecture two", chunks)
        doc_id = documents[0]["id"]
        changed = client.patch(url + "/" + doc_id,
                      json={"document_type":"lecture_transcript","lecture_number":2})
        assert changed.status_code == 200
        matches = search_chunk_matches("Main points from lecture three",
                                       main.COURSE_SOURCE_CHUNKS[course["id"]])
        assert {m["source"] for m in matches} == {"part-b.txt"}
        with patch.object(main, "_require_professor", return_value={"id":"other"}):
            assert client.patch(url + "/" + doc_id,
                json={"document_type":"material"}).status_code == 400
        assert store.list_documents(course["id"])[0]["lecture_number"] in (2,3)

def test_old_schema_migrates_without_guessing_lectures(setup):
    store, professor, course = setup
    store.save_document(course["id"],professor["id"],"old.txt","transcript",
                        b"Lecture material","Lecture material")
    with store._connect() as connection:
        connection.execute("ALTER TABLE documents DROP COLUMN lecture_number")
    store.initialize()
    doc = store.list_documents(course["id"])[0]
    assert doc["lecture_number"] is None
    assert doc["document_type"] == "transcript"
    assert store.load_course_materials(course["id"])[1]["old.txt"] == "Lecture material"

def test_chapter_never_substitutes_for_lecture():
    chunks = build_course_chunks("", {"chapter_03_revised.md": "Lecture three main points. " * 10})
    assert search_chunk_matches("List twenty points from lecture three",chunks) == []
    assert search_chunk_matches("Main points in chapter three",chunks)

@pytest.mark.parametrize("kind,number", [("lecture_transcript",None),("lecture_transcript",0),
    ("lecture_transcript",101),("lecture_transcript",True),("material",3),("syllabus",2)])
def test_invalid_labels_rejected(kind,number):
    with pytest.raises(PilotValidationError):
        PilotStore.validate_document_category(kind,number)
