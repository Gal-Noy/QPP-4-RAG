import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from querygym import retrieve_all_queries
from querygym import retrieve_all_queries_cohere
from querygym.run_llama_generator import build_messages, make_result, model_tag, parse_answer
from scripts.convert_to_ragnarok_format import convert_run, read_queries, read_run


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "nuggetizer" / "src"))
from nuggetizer.core.metrics import calculate_nugget_scores  # noqa: E402


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
