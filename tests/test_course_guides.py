import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException

import main
from knowledge import build_course_chunks
from pilot_platform import PilotStore, PilotValidationError, generate_encryption_key


class FakeMessages:
    def __init__(self, response_text):
        self.calls = []
        self.response_text = response_text

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(text=self.response_text)],
            usage=SimpleNamespace(input_tokens=30, output_tokens=12),
        )


class FakeClient:
    def __init__(self, response_text):
        self.messages = FakeMessages(response_text)


class CourseFeatureStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = PilotStore(
            Path(self.temp_dir.name),
            generate_encryption_key(),
            max_professors=5,
        )
        self.store.initialize()
        _, token = self.store.create_invitation(
            "guide.owner@northeastern.edu", "Guide Owner"
        )
        self.owner = self.store.accept_invitation(token, "a-secure-password")
        _, other_token = self.store.create_invitation(
            "other.owner@northeastern.edu", "Other Owner"
        )
        self.other = self.store.accept_invitation(
            other_token, "another-secure-password"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_feature_defaults_and_updates_preserve_course_data(self):
        course = self.store.create_course(
            self.owner["id"], "Guide Course", "TEST 2000", "Fall 2026"
        )
        self.assertEqual(course["project_builder_enabled"], 1)
        self.assertEqual(course["research_innovation_enabled"], 0)

        document = self.store.save_document(
            course["id"],
            self.owner["id"],
            "syllabus.txt",
            "syllabus",
            b"Regularization reduces overfitting.",
            "Regularization reduces overfitting.",
        )
        concept_map = {
            "Regularization": {
                "description": "A penalty that reduces overfitting.",
                "lectures": ["Course syllabus"],
                "prerequisites": [],
                "related": [],
            }
        }
        self.store.set_concept_map(course["id"], self.owner["id"], concept_map)

        updated = self.store.set_course_features(
            course["id"], self.owner["id"], False, True
        )
        self.assertEqual(updated["project_builder_enabled"], 0)
        self.assertEqual(updated["research_innovation_enabled"], 1)
        self.assertEqual(
            self.store.list_documents(course["id"], self.owner["id"])[0]["id"],
            document["id"],
        )
        self.assertEqual(
            self.store.load_course_materials(course["id"])[2], concept_map
        )

        with self.assertRaises(PilotValidationError):
            self.store.set_course_features(
                course["id"], self.other["id"], True, True
            )


class GuideChatTests(unittest.TestCase):
    def setUp(self):
        self.old_courses = main.COURSES
        self.old_prompts = main.SYSTEM_PROMPTS
        self.old_chunks = main.COURSE_SOURCE_CHUNKS
        self.old_client = main.CLIENT
        main.COURSES = {
            "guide-test": {
                "code": "TEST2000",
                "name": "Guide Test",
                "professor": "Test Professor",
                "campus": "Arlington",
                "project_builder_enabled": True,
                "research_innovation_enabled": False,
            }
        }
        main.SYSTEM_PROMPTS = {"guide-test": "Use only supplied course sources."}
        main.COURSE_SOURCE_CHUNKS = {
            "guide-test": build_course_chunks(
                "Regularization reduces overfitting by adding a penalty.", {}
            )
        }

    def tearDown(self):
        main.COURSES = self.old_courses
        main.SYSTEM_PROMPTS = self.old_prompts
        main.COURSE_SOURCE_CHUNKS = self.old_chunks
        main.CLIENT = self.old_client

    def test_project_builder_uses_guide_prompt_and_recent_topic_sources(self):
        fake_client = FakeClient(
            "Which result would you most like the project to demonstrate?"
        )
        main.CLIENT = fake_client
        history = [
            main.ChatMessage(
                role="assistant",
                content="What course topic would you like to turn into a project?",
            ),
            main.ChatMessage(role="user", content="Regularization"),
            main.ChatMessage(
                role="assistant", content="What kind of deliverable interests you?"
            ),
        ]
        response = asyncio.run(
            main.chat(
                "guide-test",
                main.ChatRequest(
                    message="I have two weeks and would like a visual demonstration.",
                    history=history,
                    mode="project_builder",
                ),
            )
        )
        payload = json.loads(response.body)

        self.assertEqual(payload["mode"], "project_builder")
        self.assertTrue(payload["materials_found"])
        self.assertEqual(payload["sources"][0]["name"], "Course syllabus (syllabus.md)")
        call = fake_client.messages.calls[0]
        self.assertIn("Active workflow: Project Builder", call["system"])
        self.assertIn("exactly one concise question", call["system"])

    def test_disabled_research_guide_is_rejected_even_if_called_directly(self):
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(
                main.chat(
                    "guide-test",
                    main.ChatRequest(
                        message="Investigate regularization",
                        mode="research_innovation",
                    ),
                )
            )
        self.assertEqual(raised.exception.status_code, 403)
        self.assertIn("not enabled", raised.exception.detail)

    def test_research_prompt_requires_novelty_caution(self):
        main.COURSES["guide-test"]["research_innovation_enabled"] = True
        fake_client = FakeClient("Which assumption would you examine first?")
        main.CLIENT = fake_client
        response = asyncio.run(
            main.chat(
                "guide-test",
                main.ChatRequest(
                    message="Investigate regularization and overfitting",
                    mode="research_innovation",
                ),
            )
        )
        payload = json.loads(response.body)
        self.assertEqual(payload["mode"], "research_innovation")
        call = fake_client.messages.calls[0]
        self.assertIn("Research and Innovation Guide", call["system"])
        self.assertIn("Never call an idea novel", call["system"])


class GuideInterfaceTests(unittest.TestCase):
    def test_student_and_faculty_controls_are_present(self):
        base = Path(main.__file__).parent
        course_html = (base / "static" / "course.html").read_text()
        faculty_html = (base / "static" / "faculty.html").read_text()

        self.assertIn("Turn this into a project", course_html)
        self.assertIn("research-innovation-button", course_html)
        self.assertIn("mode: requestMode", course_html)
        self.assertIn("Download project brief", course_html)
        self.assertIn("Save guide settings", faculty_html)
        self.assertIn("project_builder_enabled", faculty_html)
        self.assertIn("research_innovation_enabled", faculty_html)


if __name__ == "__main__":
    unittest.main()
