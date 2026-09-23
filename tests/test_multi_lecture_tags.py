from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
import main
from knowledge import search_chunk_matches
from pilot_platform import PilotValidationError
from test_lecture_labels import setup

def test_tags_upload_edit_filter_and_syllabus(setup):
    store, professor, course = setup
    with patch.object(main,"_require_pilot_store",return_value=store), \
         patch.object(main,"_require_professor",return_value=professor), \
         patch.object(main,"COURSES",{}), patch.object(main,"COURSE_SOURCE_CHUNKS",{}), \
         patch.object(main,"CONCEPT_MAPS",{}), patch.object(main,"SYSTEM_PROMPTS",{}):
        client=TestClient(main.app)
        url="/api/faculty/courses/"+course["id"]+"/documents"
        doc=client.post(url,data={"document_type":"material","lecture_numbers":"[1,2]"},
                        files={"files":("slides.txt",b"A tool is a callable function.")})
        assert doc.status_code==200,doc.text
        docid=doc.json()["documents"][0]["id"]
        assert store.list_documents(course["id"])[0]["lecture_numbers"]==[1,2]
        client.post(url,data={"document_type":"syllabus","lecture_numbers":"[]"},
                    files={"files":("syllabus.txt",b"Private unrelated syllabus fact.")})
        syllabus=next(d for d in store.list_documents(course["id"]) if d["document_type"]=="syllabus")
        assert client.patch(url+"/"+syllabus["id"],json={"document_type":"syllabus","lecture_numbers":[2]}).status_code==200
        assert client.patch(url+"/"+docid,json={"document_type":"syllabus","lecture_numbers":[]}).status_code==400
        assert client.patch(url+"/"+syllabus["id"],json={"document_type":"syllabus","lecture_numbers":[]}).status_code==200
        main.COURSES[course["id"]]["_status"]="published"
        calls=[]
        def fake(**kwargs):
            calls.append(kwargs)
            return main.ProviderResponse(text="A tool is a callable function.",input_tokens=5,output_tokens=5)
        with patch.object(main,"_call_course_model",side_effect=fake):
            response=client.post("/course/"+course["id"]+"/chat",
                                 json={"message":"What is a tool?","lecture_number":2})
        assert response.status_code==200,response.text
        assert response.json()["materials_found"]
        assert "slides.txt" in response.json()["sources"][0]["name"]
        assert "Private unrelated" not in calls[0]["system_prompt"]
        assert client.post("/course/"+course["id"]+"/chat",
                           json={"message":"What is a tool in lecture one?","lecture_number":2}).status_code==400
        response=client.post("/course/"+course["id"]+"/chat",
                             json={"message":"What is a tool?","lecture_number":3})
        assert not response.json()["materials_found"]
        assert "Lecture 3" in response.json()["response"]
        assert client.patch(url+"/"+docid,json={"document_type":"material","lecture_numbers":[2,3]}).status_code==200
        chunks=main.COURSE_SOURCE_CHUNKS[course["id"]]
        assert not search_chunk_matches("Main points from lecture one",chunks)
        assert search_chunk_matches("Main points from lecture three",chunks)
        assert search_chunk_matches("What is a tool?",chunks)

def test_existing_single_number_is_migrated(setup):
    store, professor, course=setup
    store.save_document(course["id"],professor["id"],"transcript.txt","lecture_transcript",
                        b"Recorded discussion","Recorded discussion",lecture_number=3)
    with store._connect() as c:
        c.execute("ALTER TABLE documents DROP COLUMN lecture_numbers")
    store.initialize()
    assert store.list_documents(course["id"])[0]["lecture_numbers"]==[3]

@pytest.mark.parametrize("tags", [[0],[16],["2"],[True],"1,2"])
def test_invalid_tags(setup,tags):
    store,_,_=setup
    with pytest.raises(PilotValidationError):
        store.validate_lecture_tags("material",tags)
