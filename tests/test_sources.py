import asyncio
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

import main
from knowledge import (
    build_course_chunks,
    extract_search_terms,
    search_chunk_matches,
)


class FakeMessages:
    def __init__(self, response_text="A grounded answer."):
        self.calls = []
        self.response_text = response_text

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(text=self.response_text)],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )


class FakeClient:
    def __init__(self, response_text="A grounded answer."):
        self.messages = FakeMessages(response_text)


class CourseSourceSearchTests(unittest.TestCase):
    def test_common_question_words_do_not_create_a_match(self):
        chunks = build_course_chunks(
            "This syllabus describes regression and regularization.",
            {},
        )
        self.assertEqual(search_chunk_matches("What is photosynthesis?", chunks), [])

    def test_two_term_question_needs_both_terms(self):
        chunks = build_course_chunks(
            "This syllabus mentions capital budgeting but not geography.",
            {},
        )
        self.assertEqual(search_chunk_matches("What is the capital of France?", chunks), [])

    def test_syllabus_and_transcript_names_are_preserved(self):
        chunks = build_course_chunks(
            "Regularization reduces overfitting.",
            {"week-10.docx": "Regularization adds a penalty to the loss function."},
        )
        matches = search_chunk_matches("What is regularization?", chunks)
        names = {match["display_name"] for match in matches}
        self.assertIn("Course syllabus (syllabus.md)", names)
        self.assertIn("Lecture transcript: week-10.docx", names)
        self.assertTrue(all(len(match["excerpt"]) <= 326 for match in matches))

    def test_short_follow_up_uses_previous_student_question(self):
        history = [
            main.ChatMessage(role="user", content="What is regularization?"),
            main.ChatMessage(role="assistant", content="It adds a penalty."),
        ]
        query = main._build_retrieval_query("Can you explain that more?", history)
        self.assertIn("regularization", query.lower())

    def test_greeting_has_no_search_terms(self):
        self.assertEqual(extract_search_terms("Hello, thank you"), [])

    def test_greeting_does_not_reuse_previous_question(self):
        history = [
            main.ChatMessage(role="user", content="What is regularization?"),
            main.ChatMessage(role="assistant", content="It adds a penalty."),
        ]
        query = main._build_retrieval_query("Thank you", history)
        self.assertEqual(query, "Thank you")


class ChatSourceResponseTests(unittest.TestCase):
    def setUp(self):
        main.COURSES = {
            "test": {
                "code": "TEST1000",
                "name": "Test Course",
                "professor": "Test Professor",
                "campus": "Test",
            }
        }
        main.SYSTEM_PROMPTS = {"test": "Answer only from supplied sources."}

    def test_chat_returns_source_name_and_excerpt(self):
        main.COURSE_SOURCE_CHUNKS = {
            "test": build_course_chunks(
                "Regularization reduces overfitting by adding a penalty.",
                {},
            )
        }
        fake_client = FakeClient("Regularization adds a penalty to reduce overfitting.")
        main.CLIENT = fake_client

        response = asyncio.run(main.chat(
            "test",
            main.ChatRequest(message="What is regularization?"),
        ))
        payload = json.loads(response.body)

        self.assertTrue(payload["materials_found"])
        self.assertEqual(payload["sources"][0]["name"], "Course syllabus (syllabus.md)")
        self.assertIn("Regularization", payload["sources"][0]["excerpt"])
        self.assertEqual(len(fake_client.messages.calls), 1)

    def test_chat_plainly_refuses_when_no_source_matches(self):
        main.COURSE_SOURCE_CHUNKS = {
            "test": build_course_chunks("This course covers regularization.", {})
        }
        fake_client = FakeClient()
        main.CLIENT = fake_client

        response = asyncio.run(main.chat(
            "test",
            main.ChatRequest(message="What is photosynthesis?"),
        ))
        payload = json.loads(response.body)

        self.assertFalse(payload["materials_found"])
        self.assertEqual(payload["sources"], [])
        self.assertEqual(payload["response"], main.NO_MATERIALS_RESPONSE)
        self.assertEqual(len(fake_client.messages.calls), 0)


class CoursePageTests(unittest.TestCase):
    def test_course_page_renders_returned_sources(self):
        html = (Path(main.__file__).parent / "static" / "course.html").read_text()
        self.assertIn("data.sources || []", html)
        self.assertIn("Supporting course material", html)
        self.assertIn("sourceExcerpt.textContent", html)


if __name__ == "__main__":
    unittest.main()
