import sqlite3
import tempfile
import unittest
from pathlib import Path

from pilot_platform import (
    MAX_EXTRACTED_TEXT_CHARS,
    PilotStore,
    PilotValidationError,
    extract_document_text,
    generate_encryption_key,
    normalize_email,
)


class PilotStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = PilotStore(
            Path(self.temp_dir.name),
            generate_encryption_key(),
            max_professors=5,
        )
        self.store.initialize()

    def tearDown(self):
        self.temp_dir.cleanup()

    def join_professor(self, email="w.claster@northeastern.edu", name="Bill Claster"):
        _, token = self.store.create_invitation(email, name)
        return self.store.accept_invitation(token, "a-secure-password")

    def test_only_northeastern_email_is_accepted(self):
        self.assertEqual(
            normalize_email(" W.Claster@NORTHEASTERN.EDU "),
            "w.claster@northeastern.edu",
        )
        with self.assertRaises(PilotValidationError):
            self.store.create_invitation("person@example.com", "Person")

    def test_invitation_is_one_time_and_expiring(self):
        _, token = self.store.create_invitation(
            "w.claster@northeastern.edu", "Bill Claster"
        )
        self.assertIsNotNone(self.store.invitation_details(token))
        professor = self.store.accept_invitation(token, "a-secure-password")
        self.assertEqual(professor["email"], "w.claster@northeastern.edu")
        self.assertIsNone(self.store.invitation_details(token))
        with self.assertRaises(PilotValidationError):
            self.store.accept_invitation(token, "another-secure-password")

    def test_five_slot_limit_counts_pending_invitations(self):
        tokens = []
        for index in range(5):
            _, token = self.store.create_invitation(
                f"professor{index}@northeastern.edu", f"Professor {index}"
            )
            tokens.append(token)
        with self.assertRaises(PilotValidationError):
            self.store.create_invitation(
                "professor5@northeastern.edu", "Professor 5"
            )

        # Renewing the same pending invitation does not consume another slot.
        _, replacement = self.store.create_invitation(
            "professor0@northeastern.edu", "Professor 0"
        )
        self.assertIsNotNone(self.store.invitation_details(replacement))
        self.assertIsNone(self.store.invitation_details(tokens[0]))

    def test_api_key_is_encrypted_and_can_be_removed(self):
        professor = self.join_professor()
        api_key = "sk-ant-test-secret-value-1234567890"
        updated = self.store.set_professor_api_key(professor["id"], api_key)
        self.assertTrue(updated["has_api_key"])
        self.assertEqual(updated["api_key_last_four"], "7890")
        self.assertEqual(self.store.decrypted_api_key(professor["id"]), api_key)
        self.assertNotIn(api_key.encode("utf-8"), self.store.db_path.read_bytes())

        removed = self.store.delete_professor_api_key(professor["id"])
        self.assertFalse(removed["has_api_key"])
        self.assertIsNone(self.store.decrypted_api_key(professor["id"]))

    def test_course_documents_are_scoped_to_the_owner(self):
        first = self.join_professor()
        second = self.join_professor(
            "other.professor@northeastern.edu", "Other Professor"
        )
        course = self.store.create_course(
            first["id"], "Test Course", "TEST 1000", "Fall 2026"
        )
        extracted = "Regularization reduces overfitting."
        document = self.store.save_document(
            course["id"],
            first["id"],
            "syllabus.txt",
            "syllabus",
            extracted.encode("utf-8"),
            extracted,
        )
        self.assertEqual(document["filename"], "syllabus.txt")
        syllabus, transcripts, concept_map = self.store.load_course_materials(
            course["id"]
        )
        self.assertEqual(syllabus, extracted)
        self.assertEqual(transcripts, {})
        self.assertEqual(concept_map, {})

        with self.assertRaises(PilotValidationError):
            self.store.list_documents(course["id"], second["id"])
        with self.assertRaises(PilotValidationError):
            self.store.delete_document(
                document["id"], course["id"], second["id"]
            )

    def test_sessions_and_monthly_usage(self):
        professor = self.join_professor()
        token = self.store.create_session("professor", professor["id"])
        session = self.store.session_details(token)
        self.assertEqual(session["role"], "professor")
        self.assertEqual(session["professor_id"], professor["id"])
        self.store.delete_session(token)
        self.assertIsNone(self.store.session_details(token))

        course = self.store.create_course(
            professor["id"],
            "Test Course",
            "TEST 1000",
            "Fall 2026",
            monthly_question_limit=1,
        )
        self.store.record_usage(
            course["id"],
            professor["id"],
            "faculty-concept-map",
            "test-model",
            100,
            50,
            event_type="concept_map",
        )
        self.assertEqual(self.store.remaining_questions(course["id"]), 1)
        self.store.record_usage(
            course["id"], professor["id"], "student-session", "test-model", 20, 8
        )
        usage = self.store.monthly_usage(course["id"])
        self.assertEqual(usage["questions"], 1)
        self.assertEqual(usage["concept_map_generations"], 1)
        self.assertEqual(usage["input_tokens"], 120)
        self.assertEqual(self.store.remaining_questions(course["id"]), 0)

    def test_feedback_schema_does_not_store_chat_text(self):
        professor = self.join_professor()
        course = self.store.create_course(
            professor["id"], "Test Course", "TEST 1000", "Fall 2026"
        )
        self.store.record_feedback(
            course["id"], "student-session", "down", "Needs a clearer example"
        )
        with sqlite3.connect(self.store.db_path) as connection:
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(pilot_feedback)")
            }
        self.assertNotIn("message", columns)
        self.assertNotIn("response", columns)

    def test_extracted_text_and_course_count_are_bounded(self):
        with self.assertRaises(PilotValidationError):
            extract_document_text(
                "oversized.txt",
                b"x" * (MAX_EXTRACTED_TEXT_CHARS + 1),
            )

        professor = self.join_professor()
        for index in range(10):
            self.store.create_course(
                professor["id"],
                f"Course {index}",
                f"TEST {index}",
                "Fall 2026",
            )
        with self.assertRaises(PilotValidationError):
            self.store.create_course(
                professor["id"],
                "Course 11",
                "TEST 11",
                "Fall 2026",
            )


if __name__ == "__main__":
    unittest.main()
