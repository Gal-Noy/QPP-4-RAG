import json
import os
import sys
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from querygym import retrieve_all_queries
from querygym import retrieve_all_queries_cohere
from querygym.run_llama_generator import (
    build_messages,
    load_local_env as load_llama_env,
    make_result,
    model_tag,
    parse_answer,
)
from querygym.run_rag_nuggetizer import (
    QWEN_JUDGE_MODEL,
    ensure_manifest,
    parser_for as nuggetizer_parser_for,
)
from scripts.convert_to_ragnarok_format import convert_run, read_queries, read_run


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "nuggetizer" / "src"))
from nuggetizer.core.metrics import calculate_nugget_scores  # noqa: E402
from nuggetizer.evaluation import assign_answer_file, read_records, score_assignment_file  # noqa: E402
from nuggetizer.models.constrained_nuggetizer import ConstrainedNuggetizer  # noqa: E402


class FakeStoredDocument:
    def __init__(self, raw):
        self._raw = raw

    def raw(self):
        return self._raw


class FakeSearcher:
    def search(self, _text, k):
        return [
            SimpleNamespace(docid=f"doc-{rank}", score=100 - rank)
            for rank in range(1, k + 1)
        ]

    def doc(self, docid):
        return FakeStoredDocument(json.dumps({
            "docid": docid,
            "title": f"Title {docid}",
            "segment": f"Passage text for {docid}.",
        }))


class ReproductionPipelineTests(unittest.TestCase):
    def test_all_active_key_loaders_use_repo_env_without_override(self):
        calls = []

        def fake_load_dotenv(path=None, dotenv_path=None, override=None):
            calls.append((Path(path or dotenv_path), override))
            return True

        fake_dotenv = SimpleNamespace(load_dotenv=fake_load_dotenv)
        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with patch.dict(os.environ, {"COHERE_API_KEY": "exported-key"}):
                self.assertEqual(
                    retrieve_all_queries_cohere.get_api_key(), "exported-key"
                )
            load_llama_env()
            api = import_module("nuggetizer.utils.api")
            api._load_local_env()

        self.assertEqual(calls, [(REPO / ".env", False)] * 3)
        example = (REPO / ".env.example").read_text(encoding="utf-8")
        for key in (
            "COHERE_API_KEY",
            "OPENAI_API_KEY",
            "HF_TOKEN",
            "AZURE_OPENAI_API_KEY",
        ):
            self.assertIn(f"{key}=", example)

    def test_one_query_bm25_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            topics_file = root / "topics.original.txt"
            topics_file.write_text("q1\tquestion text\n", encoding="utf-8")
            output = root / "runs"
            output.mkdir()
            ok, failed = retrieve_all_queries.process_query_file(
                FakeSearcher(), topics_file, output, 5
            )
            self.assertEqual((ok, failed), (1, 0))
            lines = (output / "run.original.txt").read_text().splitlines()
            self.assertEqual(len(lines), 5)
            self.assertEqual([int(line.split()[3]) for line in lines], [1, 2, 3, 4, 5])
            ok, failed = retrieve_all_queries.process_query_file(
                FakeSearcher(), topics_file, output, 5
            )
            self.assertEqual((ok, failed), (0, 0))
            self.assertEqual(len((output / "run.original.txt").read_text().splitlines()), 5)

    def test_cohere_embeds_queries_in_one_batch(self):
        class FakeCohere:
            def __init__(self):
                self.calls = []

            def embed(self, **kwargs):
                self.calls.append(kwargs["texts"])
                return SimpleNamespace(
                    embeddings=SimpleNamespace(float=[[1.0], [2.0], [3.0]])
                )

        class FakeFaiss:
            def search(self, embeddings, k):
                self.last_shape = embeddings.shape
                return (
                    np.asarray([[3.0, 2.0], [3.0, 2.0], [3.0, 2.0]]),
                    np.asarray([[0, 1], [2, 3], [4, 5]]),
                )

        index = SimpleNamespace(
            config={"embedding_type": "float", "model": "embed-english-v3.0"},
            co=FakeCohere(),
            index=FakeFaiss(),
        )
        results = retrieve_all_queries_cohere.search_batch(
            index,
            ["one", "two", "three"],
            2,
            fetcher=lambda _index, doc_idx: {"docid": f"d{doc_idx}", "segment": "text"},
        )
        self.assertEqual(len(index.co.calls), 1)
        self.assertEqual(index.co.calls[0], ["one", "two", "three"])
        self.assertEqual([[item["doc"]["docid"] for item in row] for row in results],
                         [["d0", "d1"], ["d2", "d3"], ["d4", "d5"]])

    def test_cohere_corpus_download_retries_transient_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            class FakeIndex:
                local_dir = temporary
                calls = 0

                def download_from_remote(self, relative_path):
                    self.calls += 1
                    if self.calls < 3:
                        raise ConnectionError("temporary DNS failure")
                    target = Path(self.local_dir) / relative_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("complete", encoding="utf-8")

            index = FakeIndex()
            with patch.object(retrieve_all_queries_cohere.time, "sleep") as sleep:
                with patch.object(retrieve_all_queries_cohere.tqdm, "write"):
                    retrieve_all_queries_cohere.download_with_retries(
                        index, "corpus/01/00101.jsonl.zst"
                    )
            self.assertEqual(index.calls, 3)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])

    def test_top5_conversion_and_llama_result_preserve_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            query_text = "the exact reformulated query"
            topics_file = root / "topics.variant.txt"
            topics_file.write_text(f"q1\t{query_text}\n", encoding="utf-8")
            run_file = root / "run.variant.txt"
            run_file.write_text(
                "\n".join(
                    f"q1 Q0 doc-{rank} {rank} {100-rank}.0 variant"
                    for rank in range(1, 8)
                ) + "\n",
                encoding="utf-8",
            )
            converted = convert_run(
                read_queries(topics_file), read_run(run_file), FakeSearcher(), k=5
            )
            self.assertEqual(len(converted), 1)
            request = converted[0]
            self.assertEqual(request["query"]["text"], query_text)
            self.assertEqual(
                [item["docid"] for item in request["candidates"]],
                [f"doc-{rank}" for rank in range(1, 6)],
            )
            messages = build_messages(request)
            for rank in range(1, 6):
                self.assertIn(f"[{rank}]", messages[1]["content"])
            result = make_result(
                request,
                "The first fact is supported [1]. The second uses two passages [2, 5].",
                "smoke",
            )
            self.assertEqual(result["topic"], query_text)
            self.assertEqual(result["references"], [f"doc-{rank}" for rank in range(1, 6)])
            self.assertEqual(result["answer"][0]["citations"], [0])
            self.assertEqual(result["answer"][1]["citations"], [1, 4])

    def test_nugget_all_and_strict_vital_metrics(self):
        metrics = calculate_nugget_scores("q1", [
            {"importance": "vital", "assignment": "support"},
            {"importance": "vital", "assignment": "partial_support"},
            {"importance": "okay", "assignment": "not_support"},
        ])
        self.assertEqual(metrics.strict_vital_score, 0.5)
        self.assertEqual(metrics.all_score, 0.5)

    def test_constrained_local_judge_returns_one_label_per_nugget(self):
        class FakeChooser:
            def choose_one_token(self, messages, choices, labels):
                self.messages = messages
                self.choices = choices
                self.labels = labels
                return ["support"] * len(messages)

        chooser = FakeChooser()
        judge = ConstrainedNuggetizer(chooser, batch_size=2)
        nuggets = [
            SimpleNamespace(text=f"fact {index}", importance="vital")
            for index in range(3)
        ]
        assigned = judge.assign("query", "answer", nuggets)
        self.assertEqual(len(assigned), 3)
        self.assertTrue(all(item.assignment == "support" for item in assigned))
        self.assertEqual(set(chooser.choices.values()), {
            "support", "partial_support", "not_support",
        })

    def test_nuggetizer_defaults_to_local_qwen_only(self):
        args = nuggetizer_parser_for(REPO).parse_args(["--nugget-file", "nuggets.jsonl"])
        self.assertEqual(args.model, "Qwen/Qwen3-4B-Instruct-2507")
        self.assertEqual(args.model, QWEN_JUDGE_MODEL)
        self.assertFalse(hasattr(args, "judge_backend"))
        self.assertFalse(hasattr(args, "use_azure_openai"))

    def test_manifest_refuses_unidentified_or_different_judge_outputs(self):
        manifest = {"judge_backend": "local-hf", "judge_model": QWEN_JUDGE_MODEL}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "eval"
            ensure_manifest(output, manifest)
            ensure_manifest(output, manifest)
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                ensure_manifest(output, {**manifest, "judge_model": "different"})

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "eval"
            output.mkdir()
            (output / "old_scores.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "without a provenance manifest"):
                ensure_manifest(output, manifest)

    def test_resumable_assignment_and_atomic_scoring(self):
        class FakeNuggetizer:
            calls = 0

            def assign(self, _query, _answer, nuggets):
                self.calls += 1
                return [
                    SimpleNamespace(
                        text=item.text,
                        importance=item.importance,
                        assignment="support",
                    )
                    for item in nuggets
                ]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nuggets = root / "nuggets.jsonl"
            nuggets.write_text(
                json.dumps({
                    "qid": "q1", "query": "question",
                    "nuggets": [{"text": "fact", "importance": "vital"}],
                }) + "\n",
                encoding="utf-8",
            )
            answers = root / "answers.json"
            answers.write_text(json.dumps([{
                "topic_id": "q1", "response_length": 1,
                "answer": [{"text": "fact", "citations": []}],
            }]), encoding="utf-8")
            assignments = root / "assignments.jsonl"
            judge = FakeNuggetizer()
            self.assertEqual(
                assign_answer_file(nuggets, answers, assignments, judge),
                (1, {"q1"}),
            )
            self.assertEqual(
                assign_answer_file(nuggets, answers, assignments, judge),
                (0, {"q1"}),
            )
            self.assertEqual(judge.calls, 1)
            scores = root / "scores.jsonl"
            aggregate = score_assignment_file(assignments, scores)
            self.assertEqual(aggregate["all_score"], 1.0)
            self.assertEqual([row["qid"] for row in read_records(scores)], ["q1", "all"])

    def test_parser_drops_out_of_range_citations(self):
        parsed = parse_answer("Supported statement [1, 9].")
        self.assertEqual(parsed, [{"text": "Supported statement.", "citations": [0]}])

    def test_smoke_model_cannot_use_experiment_tag(self):
        self.assertEqual(
            model_tag("meta-llama/Llama-3.2-1B-Instruct"),
            "llama_3_2_1b_instruct",
        )


if __name__ == "__main__":
    unittest.main()
