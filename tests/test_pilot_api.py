import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

import main
from pilot_platform import generate_encryption_key


class FakeMessages:
    def create(self, **kwargs):
        return SimpleNamespace(
            content=[SimpleNamespace(text="Regularization helps reduce overfitting.")],
            usage=SimpleNamespace(input_tokens=24, output_tokens=9),
        )


class FakeAnthropicClient:
    def __init__(self, *args, **kwargs):
        self.messages = FakeMessages()


class FakeConceptMapMessages:
    def create(self, **kwargs):
        concept_map = {
            "Regularization": {
                "description": "A penalty used to reduce overfitting.",
                "lectures": ["Course syllabus"],
                "prerequisites": ["Model fitting"],
                "related": ["Model fitting"],
            },
            "Model fitting": {
                "description": "Estimating a model from data.",
                "lectures": ["Course syllabus"],
                "prerequisites": [],
                "related": ["Regularization"],
            },
        }
        return SimpleNamespace(
            content=[SimpleNamespace(text=json.dumps(concept_map))],
            usage=SimpleNamespace(input_tokens=120, output_tokens=60),
        )


class FakeConceptMapClient:
    def __init__(self, *args, **kwargs):
        self.messages = FakeConceptMapMessages()


class PilotApiFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.environment = mock.patch.dict(
            os.environ,
            {
                "ATLAS_DATA_DIR": self.temp_dir.name,
                "ATLAS_ENCRYPTION_KEY": generate_encryption_key(),
                "ATLAS_ADMIN_PASSWORD": "test-administrator-password",
                "ATLAS_MAX_PILOT_PROFESSORS": "5",
                "ATLAS_PUBLIC_BASE_URL": "https://atlas.example.edu",
            },
        )
        self.environment.start()
        self.old_enabled = main.PILOT_ENABLED
        self.old_secure = main.SECURE_COOKIES
        self.old_store = main.PILOT_STORE
        main.PILOT_ENABLED = True
        main.SECURE_COOKIES = False
        self.client_context = TestClient(main.app)
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        main.PILOT_ENABLED = self.old_enabled
        main.SECURE_COOKIES = self.old_secure
        main.PILOT_STORE = self.old_store
        self.environment.stop()
        self.temp_dir.cleanup()

    def json_request(self, method, path, payload=None):
        return self.client.request(
            method,
            path,
            content=json.dumps(payload or {}),
            headers={"Content-Type": "application/json"},
        )

    def test_invite_join_create_upload_publish_and_limit(self):
        self.assertEqual(self.client.get("/api/pilot-admin/summary").status_code, 401)
        login = self.json_request(
            "POST",
            "/api/pilot-admin/login",
            {"password": "test-administrator-password"},
        )
        self.assertEqual(login.status_code, 200)
        self.assertIn("HttpOnly", login.headers["set-cookie"])
        self.assertIn("SameSite=strict", login.headers["set-cookie"])
        admin_page = self.client.get("/pilot-admin")
        self.assertEqual(admin_page.status_code, 200)
        self.assertEqual(admin_page.headers["cache-control"], "no-store")
        self.assertEqual(admin_page.headers["referrer-policy"], "no-referrer")
        self.assertEqual(self.client.get("/admin/upload").status_code, 200)

        invitation = self.json_request(
            "POST",
            "/api/pilot-admin/invitations",
            {
                "email": "w.claster@northeastern.edu",
                "name": "Bill Claster",
                "expires_hours": 72,
            },
        )
        self.assertEqual(invitation.status_code, 201)
        self.assertTrue(
            invitation.json()["join_url"].startswith(
                "https://atlas.example.edu/faculty/join#token="
            )
        )
        self.assertNotIn("?token=", invitation.json()["join_url"])
        token = invitation.json()["join_path"].split("token=", 1)[1]
        self.client.post("/api/pilot-admin/logout")

        joined = self.json_request(
            "POST",
            "/api/faculty/join",
            {
                "token": token,
                "name": "Bill Claster",
                "password": "a-secure-password",
            },
        )
        self.assertEqual(joined.status_code, 201)
        self.assertEqual(
            joined.json()["professor"]["email"],
            "w.claster@northeastern.edu",
        )

        key_response = self.json_request(
            "PUT",
            "/api/faculty/api-key",
            {"api_key": "sk-ant-test-secret-value-1234567890"},
        )
        self.assertEqual(key_response.status_code, 200)
        self.assertNotIn("sk-ant-test-secret-value", key_response.text)
        self.assertNotIn("api_key_encrypted", key_response.text)

        created = self.json_request(
            "POST",
            "/api/faculty/courses",
            {
                "name": "Pilot Test Course",
                "code": "TEST 1000",
                "term": "Fall 2026",
                "section": "01",
                "campus": "Arlington",
                "monthly_question_limit": 1,
            },
        )
        self.assertEqual(created.status_code, 201)
        course_id = created.json()["id"]
        self.assertTrue(course_id.startswith("c_"))
        self.assertTrue(created.json()["project_builder_enabled"])
        self.assertFalse(created.json()["research_innovation_enabled"])

        changed_features = self.json_request(
            "PUT",
            f"/api/faculty/courses/{course_id}/features",
            {
                "project_builder_enabled": False,
                "research_innovation_enabled": True,
            },
        )
        self.assertEqual(changed_features.status_code, 200)
        self.assertFalse(changed_features.json()["project_builder_enabled"])
        self.assertTrue(changed_features.json()["research_innovation_enabled"])
        restored_features = self.json_request(
            "PUT",
            f"/api/faculty/courses/{course_id}/features",
            {
                "project_builder_enabled": True,
                "research_innovation_enabled": False,
            },
        )
        self.assertEqual(restored_features.status_code, 200)
        self.assertEqual(
            self.client.get(f"/course/{course_id}/metadata").status_code,
            404,
        )

        uploaded = self.client.post(
            f"/api/faculty/courses/{course_id}/documents",
            data={"document_type": "syllabus"},
            files={
                "files": (
                    "syllabus.txt",
                    b"Regularization reduces overfitting by adding a penalty.",
                    "text/plain",
                )
            },
        )
        self.assertEqual(uploaded.status_code, 200)

        with mock.patch.object(main.anthropic, "Anthropic", FakeConceptMapClient):
            generated_map = self.client.post(
                f"/api/faculty/courses/{course_id}/concept-map"
            )
        self.assertEqual(generated_map.status_code, 200)
        self.assertEqual(generated_map.json()["concept_count"], 2)
        before_chat = self.client.get("/api/faculty/courses").json()["courses"][0]
        self.assertEqual(before_chat["usage"]["questions"], 0)
        self.assertEqual(before_chat["usage"]["concept_map_generations"], 1)
        self.assertEqual(before_chat["remaining_questions"], 1)

        published = self.json_request(
            "POST",
            f"/api/faculty/courses/{course_id}/status",
            {"status": "published"},
        )
        self.assertEqual(published.status_code, 200)
        self.assertEqual(published.json()["status"], "published")
        self.assertEqual(
            self.client.get(f"/course/{course_id}/metadata").status_code,
            200,
        )
        metadata = self.client.get(f"/course/{course_id}/metadata").json()
        self.assertTrue(metadata["project_builder_enabled"])
        self.assertFalse(metadata["research_innovation_enabled"])
        concept_map = self.client.get(f"/course/{course_id}/concept-map").json()
        self.assertEqual(len(concept_map["concepts"]), 2)
        public_ids = {
            course["id"] for course in self.client.get("/courses").json()["courses"]
        }
        self.assertNotIn(course_id, public_ids)

        with mock.patch.object(main.anthropic, "Anthropic", FakeAnthropicClient):
            answer = self.json_request(
                "POST",
                f"/course/{course_id}/chat",
                {"message": "What is regularization?", "session_id": "student-one"},
            )
        self.assertEqual(answer.status_code, 200)
        self.assertTrue(answer.json()["materials_found"])
        self.assertEqual(
            answer.json()["sources"][0]["name"],
            "Course syllabus (syllabus.txt)",
        )

        courses = self.client.get("/api/faculty/courses").json()["courses"]
        self.assertEqual(courses[0]["usage"]["questions"], 1)
        self.assertEqual(courses[0]["remaining_questions"], 0)

        with mock.patch.object(main.anthropic, "Anthropic", FakeAnthropicClient):
            limited = self.json_request(
                "POST",
                f"/course/{course_id}/chat",
                {"message": "Explain regularization", "session_id": "student-two"},
            )
        self.assertEqual(limited.status_code, 429)

        feedback = self.json_request(
            "POST",
            "/feedback",
            {
                "course_id": course_id,
                "session_id": "student-one",
                "message": "private student question",
                "response": "private generated answer",
                "rating": "up",
            },
        )
        self.assertEqual(feedback.status_code, 200)
        self.assertNotIn(b"private student question", main.PILOT_STORE.db_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
