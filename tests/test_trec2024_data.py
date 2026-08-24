import importlib.util
import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "nuggetizer" / "src"))

from scripts.prepare_trec2024_data import (  # noqa: E402
    convert_nugget_assignments,
    read_jsonl,
    read_qrels,
    read_topic_qids,
)


class Trec2024DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.topics = REPO / "querygym" / "queries" / "topics.original.txt"
        cls.qrels_path = REPO / "data" / "2024-retrieval-qrels.txt"
        cls.source_path = REPO / "data" / "nugget_assignment.20241218.jsonl"
        cls.derived_path = (
            REPO / "data"
            / "hr_scored_nist_nuggets_20241218_rag24.test_qrels_nist.jsonl"
        )

    def test_56_qid_alignment(self):
        topic_qids = read_topic_qids(self.topics)
        qrel_qids = set(read_qrels(self.qrels_path))
        source_qids = {str(record["qid"]) for record in read_jsonl(self.source_path)}
        derived_qids = {str(record["qid"]) for record in read_jsonl(self.derived_path)}
        self.assertEqual(len(topic_qids), 56)
        self.assertTrue(set(topic_qids).issubset(qrel_qids))
        self.assertEqual(source_qids, set(topic_qids))
        self.assertEqual(derived_qids, set(topic_qids))

    def test_qrels_parsing(self):
        qrels = read_qrels(self.qrels_path)
        self.assertEqual(len(qrels), 86)
        self.assertEqual(
            qrels["2024-105741"]["msmarco_v2.1_doc_00_125364462#6_229054655"],
            0,
        )
        fake_pytrec_eval = types.ModuleType("pytrec_eval")
        evaluator_path = REPO / "querygym" / "evaluate_retrieval_per_query.py"
        spec = importlib.util.spec_from_file_location(
            "evaluate_retrieval_per_query_compat", evaluator_path
        )
        evaluator_module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"pytrec_eval": fake_pytrec_eval}):
            spec.loader.exec_module(evaluator_module)
        self.assertEqual(evaluator_module.load_qrels(self.qrels_path), qrels)
        with tempfile.TemporaryDirectory() as temporary:
            malformed = Path(temporary) / "qrels.txt"
            malformed.write_text("qid 0 doc not-an-int\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid relevance"):
                read_qrels(malformed)

    def test_nugget_conversion_and_parsing(self):
        topic_qids = read_topic_qids(self.topics)
        converted = convert_nugget_assignments(
            read_jsonl(self.source_path), topic_qids
        )
        derived = read_jsonl(self.derived_path)
        self.assertEqual(converted, derived)
        self.assertEqual(len(derived), 56)
        for record in derived:
            self.assertEqual(set(record), {"qid", "query", "nuggets"})
            self.assertTrue(record["nuggets"])
            for nugget in record["nuggets"]:
                self.assertEqual(set(nugget), {"text", "importance"})
                self.assertIn(nugget["importance"], {"vital", "okay"})

    def test_derived_records_are_evaluator_compatible(self):
        fake_model_module = types.ModuleType("nuggetizer.models.nuggetizer")
        fake_model_module.Nuggetizer = object
        script_path = REPO / "nuggetizer" / "scripts" / "assign_nuggets.py"
        spec = importlib.util.spec_from_file_location("assign_nuggets_compat", script_path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules, {"nuggetizer.models.nuggetizer": fake_model_module}
        ):
            spec.loader.exec_module(module)

        record = module.read_records(str(self.derived_path))[0]

        class FakeEvaluator:
            def assign(self, query, answer, nuggets):
                self.received = (query, answer, nuggets)
                return [
                    SimpleNamespace(
                        text=nugget.text,
                        importance=nugget.importance,
                        assignment="support",
                    )
                    for nugget in nuggets
                ]

        evaluator = FakeEvaluator()
        answer = {
            "answer": [{"text": "A smoke answer."}],
            "response_length": 3,
        }
        output = module.process_record(
            answer, record, "smoke", evaluator, logging.getLogger("test")
        )
        self.assertEqual(output["qid"], record["qid"])
        self.assertEqual(len(output["nuggets"]), len(record["nuggets"]))
        self.assertTrue(all(nugget["assignment"] == "support" for nugget in output["nuggets"]))


if __name__ == "__main__":
    unittest.main()
