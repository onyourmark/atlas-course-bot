import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

import main
from pilot_platform import PilotStore, PilotValidationError, generate_encryption_key


CONCEPT_MAP = {
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


class FakeOpenAIResponses:
    calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = (
            json.dumps(CONCEPT_MAP)
            if kwargs["max_output_tokens"] == 4096
            else "OpenAI says regularization reduces overfitting."
        )
        return SimpleNamespace(
            output_text=text,
            usage=SimpleNamespace(input_tokens=31, output_tokens=12),
        )


class FakeOpenAIClient:
    def __init__(self, *args, **kwargs):
        self.responses = FakeOpenAIResponses()


class FakeAnthropicMessages:
    calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(text="Claude says regularization reduces overfitting.")],
            usage=SimpleNamespace(input_tokens=21, output_tokens=8),
        )


class FakeAnthropicClient:
    def __init__(self, *args, **kwargs):
        self.messages = FakeAnthropicMessages()


class ProviderStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = PilotStore(
            self.temp_dir.name,
            generate_encryption_key(),
            max_professors=5,
        )
        self.store.initialize()
        _, token = self.store.create_invitation(
            "w.claster@northeastern.edu", "Bill Claster"
        )
        self.professor = self.store.accept_invitation(token, "a-secure-password")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_provider_keys_are_separate_and_encrypted(self):
        anthropic_key = "sk-ant-test-secret-value-1234567890"
        openai_key = "sk-proj-test-secret-value-0987654321"
        self.store.set_professor_api_key(
            self.professor["id"], anthropic_key, "anthropic"
        )
        updated = self.store.set_professor_api_key(
            self.professor["id"], openai_key, "openai"
        )

        self.assertTrue(updated["api_keys"]["anthropic"]["has_key"])
        self.assertTrue(updated["api_keys"]["openai"]["has_key"])
        self.assertEqual(
            self.store.decrypted_api_key(self.professor["id"], "anthropic"),
            anthropic_key,
        )
        self.assertEqual(
            self.store.decrypted_api_key(self.professor["id"], "openai"),
            openai_key,
        )
        database = self.store.db_path.read_bytes()
        self.assertNotIn(anthropic_key.encode(), database)
        self.assertNotIn(openai_key.encode(), database)

    def test_model_switch_requires_provider_key_and_records_history(self):
        self.store.set_professor_api_key(
            self.professor["id"],
            "sk-ant-test-secret-value-1234567890",
            "anthropic",
        )
        course = self.store.create_course(
            self.professor["id"],
            "Test Course",
            "TEST 1000",
            "Fall 2026",
            provider="anthropic",
            model="claude-sonnet-5",
        )
        self.store.set_concept_map(course["id"], self.professor["id"], CONCEPT_MAP)

        with self.assertRaises(PilotValidationError):
            self.store.set_course_model(
                course["id"], self.professor["id"], "openai", "gpt-5.6-luna"
            )

        self.store.set_professor_api_key(
            self.professor["id"],
            "sk-proj-test-secret-value-0987654321",
            "openai",
        )
        updated = self.store.set_course_model(
            course["id"], self.professor["id"], "openai", "gpt-5.6-luna"
        )
        self.assertEqual(updated["provider"], "openai")
        self.assertEqual(updated["model"], "gpt-5.6-luna")
        _, _, concept_map = self.store.load_course_materials(course["id"])
        self.assertEqual(concept_map, CONCEPT_MAP)
        history = self.store.course_model_history(course["id"], self.professor["id"])
        self.assertEqual(history[0]["old_model"], "claude-sonnet-5")
        self.assertEqual(history[0]["new_model"], "gpt-5.6-luna")

    def test_existing_pilot_database_is_upgraded_without_changing_course_model(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "atlas_pilot.sqlite3"
            with sqlite3.connect(db_path) as connection:
                connection.executescript(
                    """
                    CREATE TABLE professors (
                        id TEXT PRIMARY KEY,
                        email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                        name TEXT NOT NULL,
                        password_hash TEXT NOT NULL,
                        api_key_encrypted TEXT,
                        api_key_last_four TEXT,
                        is_active INTEGER NOT NULL DEFAULT 1,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE courses (
                        id TEXT PRIMARY KEY,
                        owner_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        code TEXT NOT NULL,
                        section TEXT NOT NULL DEFAULT '',
                        term TEXT NOT NULL,
                        campus TEXT NOT NULL DEFAULT 'Arlington',
                        status TEXT NOT NULL DEFAULT 'draft',
                        monthly_question_limit INTEGER NOT NULL DEFAULT 500,
                        concept_map_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (owner_id) REFERENCES professors(id)
                    );
                    CREATE TABLE usage_events (
                        id TEXT PRIMARY KEY,
                        course_id TEXT NOT NULL,
                        professor_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        model TEXT NOT NULL,
                        event_type TEXT NOT NULL DEFAULT 'student_question',
                        input_tokens INTEGER NOT NULL,
                        output_tokens INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE,
                        FOREIGN KEY (professor_id) REFERENCES professors(id)
                    );
                    """
                )
                connection.execute(
                    """
                    INSERT INTO professors
                        (id, email, name, password_hash, is_active, created_at)
                    VALUES ('p_old', 'old@northeastern.edu', 'Old Professor',
                            'unused', 1, '2026-08-01T00:00:00+00:00')
                    """
                )
                connection.execute(
                    """
                    INSERT INTO courses
                        (id, owner_id, name, code, term, status,
                         monthly_question_limit, created_at, updated_at)
                    VALUES ('c_old', 'p_old', 'Old Course', 'TEST 1000', 'Fall 2026',
                            'published', 500, '2026-08-01T00:00:00+00:00',
                            '2026-08-01T00:00:00+00:00')
                    """
                )
                connection.execute(
                    """
                    INSERT INTO usage_events
                        (id, course_id, professor_id, session_id, model,
                         input_tokens, output_tokens, created_at)
                    VALUES ('u_old', 'c_old', 'p_old', 'student-old',
                            'claude-sonnet-4-6', 20, 8, '2099-08-01T00:00:00+00:00')
                    """
                )

            store = PilotStore(directory, generate_encryption_key(), max_professors=5)
            store.initialize()
            course = store.get_course("c_old")
            professor = store.get_professor("p_old")
            self.assertEqual(course["provider"], "anthropic")
            self.assertEqual(course["model"], "claude-sonnet-4-6")
            self.assertFalse(professor["api_keys"]["openai"]["has_key"])
            self.assertEqual(
                store.monthly_usage("c_old")["by_model"][0]["provider"],
                "anthropic",
            )


class ProviderApiTests(unittest.TestCase):
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
        self._join_professor()
        FakeOpenAIResponses.calls = []
        FakeAnthropicMessages.calls = []

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

    def _join_professor(self):
        self.json_request(
            "POST",
            "/api/pilot-admin/login",
            {"password": "test-administrator-password"},
        )
        invitation = self.json_request(
            "POST",
            "/api/pilot-admin/invitations",
            {
                "email": "w.claster@northeastern.edu",
                "name": "Bill Claster",
                "expires_hours": 72,
            },
        ).json()
        token = invitation["join_path"].split("token=", 1)[1]
        self.client.post("/api/pilot-admin/logout")
        self.json_request(
            "POST",
            "/api/faculty/join",
            {
                "token": token,
                "name": "Bill Claster",
                "password": "a-secure-password",
            },
        )

    def _save_keys(self):
        for provider, key in (
            ("anthropic", "sk-ant-test-secret-value-1234567890"),
            ("openai", "sk-proj-test-secret-value-0987654321"),
        ):
            response = self.json_request(
                "PUT", f"/api/faculty/api-keys/{provider}", {"api_key": key}
            )
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("test-secret-value", response.text)

    def test_openai_course_can_switch_to_anthropic_without_rebuilding_map(self):
        self._save_keys()
        me = self.client.get("/api/faculty/me").json()
        self.assertEqual([item["id"] for item in me["model_catalog"]], ["anthropic", "openai"])

        created = self.json_request(
            "POST",
            "/api/faculty/courses",
            {
                "name": "Provider Test Course",
                "code": "TEST 2000",
                "term": "Fall 2026",
                "monthly_question_limit": 10,
                "provider": "openai",
                "model": "gpt-5.6-luna",
            },
        )
        self.assertEqual(created.status_code, 201)
        course_id = created.json()["id"]

        self.client.post(
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
        with mock.patch.object(main.openai, "OpenAI", FakeOpenAIClient):
            generated = self.client.post(
                f"/api/faculty/courses/{course_id}/concept-map"
            )
        self.assertEqual(generated.status_code, 200)
        self.assertFalse(FakeOpenAIResponses.calls[0]["store"])

        self.json_request(
            "POST",
            f"/api/faculty/courses/{course_id}/status",
            {"status": "published"},
        )
        with mock.patch.object(main.openai, "OpenAI", FakeOpenAIClient):
            openai_answer = self.json_request(
                "POST",
                f"/course/{course_id}/chat",
                {"message": "What is regularization?", "session_id": "student-one"},
            )
        self.assertIn("OpenAI says", openai_answer.json()["response"])

        switched = self.json_request(
            "PUT",
            f"/api/faculty/courses/{course_id}/model",
            {"provider": "anthropic", "model": "claude-sonnet-5"},
        )
        self.assertEqual(switched.status_code, 200)
        self.assertEqual(switched.json()["concept_count"], 2)
        self.assertEqual(switched.json()["model_history"][0]["old_provider"], "openai")

        with mock.patch.object(main.anthropic, "Anthropic", FakeAnthropicClient):
            anthropic_answer = self.json_request(
                "POST",
                f"/course/{course_id}/chat",
                {"message": "Explain regularization", "session_id": "student-two"},
            )
        self.assertIn("Claude says", anthropic_answer.json()["response"])

        course = self.client.get("/api/faculty/courses").json()["courses"][0]
        providers = {item["provider"] for item in course["usage"]["by_model"]}
        self.assertEqual(providers, {"anthropic", "openai"})


if __name__ == "__main__":
    unittest.main()
