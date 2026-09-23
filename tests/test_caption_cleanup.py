from transcript_text import clean_caption_text
from pilot_platform import extract_document_text, PilotStore, generate_encryption_key

CAPTIONS = """WEBVTT

1b7cdb7e-695c-4c37-8325-b9c19deb858d/14008-0
03:05:36.103 --> 03:05:40.103
<v Claster, William>The goal,
the goal states what the loop is trying</v>

1b7cdb7e-695c-4c37-8325-b9c19deb858d/14008-1
03:05:40.103 --> 03:05:44.423
<v Claster, William>to accomplish &amp; check.</v>

1b7cdb7e-695c-4c37-8325-b9c19deb858d/14009-0
03:05:45.103 --> 03:05:47.423
<v Student>Can it stop?</v>
"""

def test_caption_cleanup_preserves_words_speakers_and_reference_time():
    clean = clean_caption_text(CAPTIONS)
    assert clean == ("[03:05:36.103] Claster, William: The goal, the goal states what the loop is trying to accomplish & check."
                     "\n\n[03:05:45.103] Student: Can it stop?")
    assert clean_caption_text(clean) == clean
    assert extract_document_text("lecture.txt", CAPTIONS.encode()) == clean

def test_plain_text_code_and_timestamps_are_not_stripped():
    text = "Example:\n03:05:36.103 --> 03:05:40.103\n<v> is a tag.\n123\nA -> B"
    assert clean_caption_text(text) == text
    assert clean_caption_text("A normal Word transcript.\nInstructor 3:20\nHello") == "A normal Word transcript.\nInstructor 3:20\nHello"

def test_srt_and_periodic_timestamps():
    text = "1\n00:00:01,000 --> 00:00:03,000\nFirst.\n\n2\n00:01:05,000 --> 00:01:07,000\nSecond."
    assert clean_caption_text(text) == "[00:00:01.000] First.\n\n[00:01:05.000] Second."

def test_existing_caption_uploads_are_cleaned_without_changing_originals(tmp_path):
    store = PilotStore(tmp_path, generate_encryption_key())
    store.initialize()
    _, token = store.create_invitation("test@northeastern.edu","Instructor")
    professor = store.accept_invitation(token,"a-secure-password")
    course = store.create_course(professor["id"],"Test","TEST1","Fall 2026")
    record = store.save_document(course["id"],professor["id"],"old.txt","transcript",CAPTIONS.encode(),CAPTIONS)
    before = (tmp_path / record["stored_path"]).read_bytes()
    text = store.load_course_materials(course["id"])[1]["old.txt"]
    assert text == clean_caption_text(CAPTIONS)
    assert (tmp_path / record["stored_path"]).read_bytes() == before
    assert (tmp_path / record["extracted_path"]).read_text() == CAPTIONS
