#!/usr/bin/env python3
"""Generate Ragnarok/TREC-RAG answers locally with Llama-3.2-3B-Instruct."""

import argparse
import json
import re
from pathlib import Path


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_TAG = "llama_3_2_3b_instruct"
CHATQA_SYSTEM = (
    "This is a chat between a user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's "
    "questions based on the context. The assistant should also indicate when "
    "the answer cannot be found in the context."
)


def load_local_env() -> None:
    """Load repository-root .env without overriding exported variables."""
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)


def load_json_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def passage_text(candidate: dict) -> str:
    doc = candidate.get("doc", {})
    if isinstance(doc, str):
        return doc
    text = doc.get("segment") or doc.get("text") or doc.get("contents") or doc.get("passage")
    if not text:
        raise ValueError(f"Candidate {candidate.get('docid')} has no passage text")
    if doc.get("title"):
        text = f"Title: {doc['title']} Content: {text}"
    # Ragnarok avoids interpreting bracketed numbers inside passages as citations.
    return re.sub(r"\[(\d+)\]", r"(\1)", " ".join(str(text).split()))


def validate_request(record: dict) -> None:
    query = record.get("query", {})
    if not query.get("qid") or not isinstance(query.get("text"), str):
        raise ValueError("Every request needs query.qid and query.text")
    candidates = record.get("candidates", [])
    if len(candidates) != 5:
        raise ValueError(
            f"{query.get('qid')}: expected exactly 5 prepared candidates, got {len(candidates)}"
        )
    docids = [candidate.get("docid") for candidate in candidates]
    if any(not docid for docid in docids) or len(set(docids)) != 5:
        raise ValueError(f"{query.get('qid')}: candidate docids are missing or duplicated")
    for candidate in candidates:
        passage_text(candidate)


def build_messages(record: dict) -> list[dict[str, str]]:
    validate_request(record)
    context = "\n\n".join(
        f"[{rank}] {passage_text(candidate)}"
        for rank, candidate in enumerate(record["candidates"], 1)
    )
    question = record["query"]["text"]
    user = (
        f"{context}\n\n"
        "Please give a full and complete answer for the question using only the "
        "context above. Cite supporting passages after each sentence with their "
        "bracketed numbers, for example [1] or [1, 3].\n\n"
        f"Question: {question}"
    )
    return [
        {"role": "system", "content": CHATQA_SYSTEM},
        {"role": "user", "content": user},
    ]


def parse_answer(text: str, reference_count: int = 5) -> list[dict]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    chunks = re.split(r"\n+|(?<=[.!?])\s+(?=(?:[-*]\s*)?[A-Z0-9])", text)
    answer = []
    for chunk in chunks:
        chunk = re.sub(r"^[-*]\s+", "", chunk.strip())
        if not chunk:
            continue
        citations = []
        for group in re.findall(r"\[([\d,\s]+)\]", chunk):
            citations.extend(int(value) - 1 for value in re.findall(r"\d+", group))
        citations = sorted({value for value in citations if 0 <= value < reference_count})
        sentence = re.sub(r"\s*\[[\d,\s]+\]", "", chunk).strip()
        if sentence:
            answer.append({"text": sentence, "citations": citations})
    if not answer and text:
        answer = [{"text": text, "citations": []}]
    return answer


def make_result(record: dict, generated_text: str, run_id: str) -> dict:
    validate_request(record)
    references = [candidate["docid"] for candidate in record["candidates"]]
    answer = parse_answer(generated_text, len(references))
    answer_text = " ".join(item["text"] for item in answer)
    return {
        "run_id": run_id,
        "topic_id": record["query"]["qid"],
        # Preserve prepared query.text verbatim; never reload original/reformulated topics here.
        "topic": record["query"]["text"],
        "references": references,
        "response_length": len(answer_text.split()),
        "answer": answer,
    }


def write_checkpoint(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def model_tag(model_name: str) -> str:
    if model_name == MODEL_NAME:
        return MODEL_TAG
    return re.sub(r"[^a-z0-9]+", "_", model_name.rsplit("/", 1)[-1].lower()).strip("_")


def result_name(input_file: Path, generator_tag: str = MODEL_TAG) -> str:
    stem = input_file.stem.replace("ragnarok_format_", "")
    return f"rag_results_{stem}_{generator_tag}_top5.json"


class LocalLlamaGenerator:
    def __init__(self, model_name: str, device: str, dtype: str,
                 local_files_only: bool, load_in_4bit: bool):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        kwargs = {"local_files_only": local_files_only}
        if dtype != "auto":
            kwargs["dtype"] = getattr(torch, dtype)
        else:
            kwargs["dtype"] = "auto"
        if load_in_4bit:
            kwargs["load_in_4bit"] = True
            kwargs["device_map"] = "auto"
        elif device == "auto":
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if device != "auto" and not load_in_4bit:
            self.model.to(device)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def generate(self, records: list[dict], max_new_tokens: int,
                 context_size: int) -> list[str]:
        prompts = [
            self.tokenizer.apply_chat_template(
                build_messages(record), tokenize=False, add_generation_prompt=True
            )
            for record in records
        ]
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        too_long = [
            record["query"]["qid"]
            for record, length in zip(records, input_lengths)
            if length + max_new_tokens > context_size
        ]
        if too_long:
            raise ValueError(
                f"Prompts exceed --context-size without truncation: {', '.join(too_long)}"
            )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_width = encoded["input_ids"].shape[1]
        return self.tokenizer.batch_decode(
            output[:, prompt_width:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


def process_file(input_file: Path, output_file: Path, generator,
                 batch_size: int, max_new_tokens: int, context_size: int,
                 qid: str | None, max_records: int | None,
                 generator_tag: str = MODEL_TAG) -> tuple[int, int]:
    requests = load_json_records(input_file)
    for record in requests:
        validate_request(record)
    if qid:
        requests = [record for record in requests if record["query"]["qid"] == qid]
        if not requests:
            raise ValueError(f"QID {qid!r} is absent from {input_file}")
    if max_records is not None:
        requests = requests[:max_records]

    existing = load_json_records(output_file) if output_file.exists() else []
    done = {record["topic_id"] for record in existing}
    pending = [record for record in requests if record["query"]["qid"] not in done]
    run_id = (
        f"{input_file.stem.replace('ragnarok_format_', '')}_{generator_tag}_top5"
    )
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        generated = generator.generate(batch, max_new_tokens, context_size)
        existing.extend(
            make_result(record, text, run_id)
            for record, text in zip(batch, generated)
        )
        write_checkpoint(output_file, existing)
        print(f"{input_file.name}: checkpointed {len(existing)}/{len(requests)}")
    return len(pending), len(requests)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--context-size", type=int, default=8192)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--qid")
    parser.add_argument("--max-records", type=int)
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Validate Top-5 files and output mapping without loading the model",
    )
    args = parser.parse_args()
    load_local_env()
    if len(args.input_dirs) != len(args.output_dirs):
        parser.error("--input-dirs and --output-dirs must have equal lengths")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.model != MODEL_NAME:
        print(
            f"WARNING: {args.model} is a code-path smoke model; experimental runs "
            f"must use {MODEL_NAME}"
        )

    generator_tag = model_tag(args.model)
    work = []
    for input_dir, output_dir in zip(args.input_dirs, args.output_dirs):
        files = sorted(input_dir.glob("ragnarok_format_run.*.json"))
        if args.max_files is not None:
            files = files[:args.max_files]
        work.extend((path, output_dir / result_name(path, generator_tag)) for path in files)
    if not work:
        parser.error("No Ragnarok prepared files found")

    if args.validate_only:
        count = 0
        for input_file, _ in work:
            records = load_json_records(input_file)
            for record in records:
                validate_request(record)
                count += 1
        print(f"Validated {len(work)} files and {count} Top-5 requests")
        return 0

    generator = LocalLlamaGenerator(
        args.model, args.device, args.dtype, args.local_files_only, args.load_in_4bit
    )
    generated = expected = 0
    for input_file, output_file in work:
        new, total = process_file(
            input_file, output_file, generator, args.batch_size,
            args.max_new_tokens, args.context_size, args.qid, args.max_records,
            generator_tag,
        )
        generated += new
        expected += total
    print(f"Llama generation complete: {generated} new answers; {expected} selected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
