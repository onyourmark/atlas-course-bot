import json
import unittest

from concept_maps import build_concept_map_prompt, parse_concept_map_response


class ConceptMapTests(unittest.TestCase):
    def test_prompt_is_bounded_and_samples_sources(self):
        prompt = build_concept_map_prompt(
            {"code": "TEST 1000", "name": "Test Course"},
            "syllabus topic " * 10000,
            {
                "lecture-one.txt": "first lecture topic " * 10000,
                "lecture-two.txt": "second lecture topic " * 10000,
            },
        )
        self.assertLessEqual(len(prompt), 63000)
        self.assertIn("Course syllabus", prompt)
        self.assertIn("lecture-one.txt", prompt)
        self.assertIn("lecture-two.txt", prompt)

    def test_response_is_normalized_and_relationships_are_validated(self):
        raw = {
            "Regression": {
                "description": "A predictive method.",
                "lectures": ["Lecture 1"],
                "prerequisites": ["Linear Algebra", "Invented Prerequisite"],
                "related": ["Linear Algebra"],
            },
            "Linear Algebra": {
                "description": "Vectors and matrices.",
                "lectures": ["Lecture 1"],
                "prerequisites": [],
                "related": ["Regression"],
            },
        }
        concept_map = parse_concept_map_response(
            "```json\n" + json.dumps(raw) + "\n```",
            "course-test",
        )
        self.assertEqual(concept_map["_meta"]["total_concepts"], 2)
        self.assertEqual(
            concept_map["Regression"]["prerequisites"],
            ["Linear Algebra"],
        )
        self.assertEqual(concept_map["_meta"]["course_id"], "course-test")


if __name__ == "__main__":
    unittest.main()
