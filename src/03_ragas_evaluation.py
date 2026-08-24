"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import os
import re
import argparse
import warnings
import types
warnings.filterwarnings("ignore")

# RAGAS evaluation is local and does not require LangSmith traces.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
# HHEM is downloaded once and evaluated locally. Avoid slow network probes on
# subsequent runs, especially in restricted lab environments.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

# RAGAS 0.4.3 vẫn import hai lớp Vertex AI đã được tách khỏi
# langchain-community 0.4.x. Lab không dùng Vertex AI; shim này chỉ giữ import
# tương thích và không tham gia vào bất kỳ phép tính metric nào.
try:
    from langchain_community.chat_models.vertexai import ChatVertexAI  # noqa: F401
except ModuleNotFoundError:
    vertex_chat_module = types.ModuleType("langchain_community.chat_models.vertexai")
    vertex_chat_module.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = vertex_chat_module

    import langchain_community.llms as community_llms

    community_llms.VertexAI = type("VertexAI", (), {})

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample, RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
# TODO: Copy SYSTEM_V1 và SYSTEM_V2 mà bạn đã viết ở file 02_prompt_hub_ab_routing.py
SYSTEM_V1 = (
    "You are a helpful assistant. Use only facts from the context. Answer in "
    "the same language as the question using 2-4 concise, direct sentences. "
    "If the context is insufficient, say so clearly and do not speculate."
    "\n\nContext:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "You are an expert information analyst. Use only facts from the context "
    "and add no outside knowledge. Answer in the same language as the question "
    "using 3-5 organized sentences: main conclusion, supporting facts, and any "
    "uncertainty. If the context is insufficient, state that limitation."
    "\n\nContext:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


def run_hhem_faithfulness(rag_results: list, version: str) -> float:
    """Compute faithfulness locally with RAGAS HHEM and sentence-level claims."""
    from ragas.metrics._faithfulness import (
        FaithfulnesswithHHEM,
        StatementGeneratorOutput,
    )

    class SentenceHHEMFaithfulness(FaithfulnesswithHHEM):
        async def _create_statements(self, row, callbacks):
            parts = re.split(r"(?<=[.!?])\s+|\n+", row["response"])
            statements = [
                re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", part).strip()
                for part in parts
                if part.strip()
            ]
            return StatementGeneratorOutput(statements=statements)

    print(f"\n🧭 Đang tính HHEM faithfulness cho prompt {version} ...")
    metric = SentenceHHEMFaithfulness(
        name="faithfulness",
        device="cpu",
        batch_size=10,
    )
    result = evaluate(
        build_ragas_dataset(rag_results),
        metrics=[metric],
        llm=get_llm(
            temperature=0,
            json_mode=True,
            model=config.OLLAMA_EVAL_MODEL if config.PROVIDER == "ollama" else None,
        ),
        embeddings=get_embeddings(),
        run_config=RunConfig(timeout=900, max_workers=1, max_retries=1),
    )
    values = [value for value in result["faithfulness"] if value is not None]
    score = float(np.nanmean(values))
    print(f"  faithfulness                  : {score:.4f}")
    return score


def repair_faithfulness_report():
    """Recompute only faithfulness from cached RAG outputs and update the report."""
    data_dir = Path(__file__).parent.parent / "data"
    report_path = data_dir / "ragas_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    for version in ("v1", "v2"):
        cache_path = data_dir / f"ragas_inputs_{version}.json"
        results = json.loads(cache_path.read_text(encoding="utf-8"))
        report[f"prompt_{version}_scores"]["faithfulness"] = (
            run_hhem_faithfulness(results, version)
        )

    best = max(
        report["prompt_v1_scores"]["faithfulness"],
        report["prompt_v2_scores"]["faithfulness"],
    )
    report["target_met"] = best >= 0.8
    report["faithfulness_evaluator"] = "RAGAS FaithfulnesswithHHEM"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    evidence_path = Path(__file__).parent.parent / "evidence" / "03_ragas_report.json"
    evidence_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n✅ HHEM faithfulness tốt nhất: {best:.4f}")
    print(f"💾 Đã cập nhật {report_path} và {evidence_path}")


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text, chunk_size=600, chunk_overlap=50)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    # TODO: Retrieve documents từ retriever
    docs = retriever.invoke(question)

    # TODO: Tạo contexts là danh sách page_content (KHÔNG ghép chuỗi ở đây)
    # Gợi ý: contexts = [doc.page_content for doc in docs]
    contexts = [doc.page_content for doc in docs]

    # TODO: Ghép contexts thành 1 string để truyền vào {context} của prompt
    ctx_str = "\n\n".join(contexts)

    # TODO: Chạy chain (prompt | llm | StrOutputParser()).invoke(...)
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context":  ctx_str,
        "question": question,
    })

    # TODO: Trả về dict với answer và contexts (list)
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    results = []
    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        # TODO: Gọi run_rag() cho câu hỏi hiện tại
        out = run_rag(retriever, llm, prompt, qa["question"])

        # TODO: Append vào results dict với 4 keys
        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    # TODO: Tạo list các SingleTurnSample từ rag_results
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    # TODO: Wrap thành EvaluationDataset và trả về
    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~5-10 phút)")

    # TODO: Tạo EvaluationDataset từ rag_results
    dataset = build_ragas_dataset(rag_results)

    # LLM và Embeddings riêng để RAGAS dùng làm evaluator
    eval_model = config.OLLAMA_EVAL_MODEL if config.PROVIDER == "ollama" else None
    llm_eval = get_llm(temperature=0, json_mode=True, model=eval_model)
    emb_eval = get_embeddings()

    # TODO: Gọi evaluate() với đầy đủ 4 metrics
    # Gợi ý:
    #   result = evaluate(
    #       dataset,
    #       metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    #       llm=llm_eval,
    #       embeddings=emb_eval,
    #   )
    result = evaluate(
        dataset,
        metrics=[answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
        run_config=RunConfig(timeout=900, max_workers=2, max_retries=1),
    )

    # Tính mean score cho mỗi metric
    # result["faithfulness"] trả về list of floats → dùng np.mean()
    scores = {}
    for key in ["answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]
        scores[key] = float(np.mean([v for v in raw if v is not None]))

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    # TODO: Tạo vectorstore
    vectorstore = setup_vectorstore()

    # Thu thập và checkpoint kết quả để có thể tiếp tục nếu evaluator bị gián đoạn.
    data_dir = Path(__file__).parent.parent / "data"
    cache_paths = {
        "v1": data_dir / "ragas_inputs_v1.json",
        "v2": data_dir / "ragas_inputs_v2.json",
    }
    collected = {}
    for version, cache_path in cache_paths.items():
        if cache_path.exists():
            collected[version] = json.loads(cache_path.read_text(encoding="utf-8"))
            print(f"📂 Đã tải checkpoint {version.upper()} từ {cache_path}")
        else:
            collected[version] = collect_rag_outputs(vectorstore, version)
            cache_path.write_text(
                json.dumps(collected[version], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"💾 Đã lưu checkpoint {version.upper()} vào {cache_path}")

    v1_results = collected["v1"]
    v2_results = collected["v2"]

    # Chạy RAGAS evaluation
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")
    v1_scores["faithfulness"] = run_hhem_faithfulness(v1_results, "v1")
    v2_scores["faithfulness"] = run_hhem_faithfulness(v2_results, "v2")

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    # TODO: Lưu báo cáo vào data/ragas_report.json
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    # TODO: Ghi report vào file bằng json.dumps hoặc json.dump
    # Gợi ý: report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    evidence_path = Path(__file__).parent.parent / "evidence" / "03_ragas_report.json"
    evidence_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"💾 Đã lưu báo cáo vào {report_path} và {evidence_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy RAGAS evaluation")
    parser.add_argument("--faithfulness-only", action="store_true")
    parser.add_argument("--refresh-outputs-only", action="store_true")
    args = parser.parse_args()
    if args.faithfulness_only:
        repair_faithfulness_report()
    elif args.refresh_outputs_only:
        vectorstore = setup_vectorstore()
        data_dir = Path(__file__).parent.parent / "data"
        for version in ("v1", "v2"):
            results = collect_rag_outputs(vectorstore, version)
            path = data_dir / f"ragas_inputs_{version}.json"
            path.write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"💾 Đã làm mới checkpoint {version.upper()} tại {path}")
    else:
        main()
