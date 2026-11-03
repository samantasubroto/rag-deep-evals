# eval_retriever.py
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from src.reranker import RerankingRetriever
from src.generator import generate
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = "goldens/retriever_goldens.json"
JUDGE_MODEL = "gpt-4o-mini"   # NOTE: differs from the other evals (gpt-4o-mini)
THRESHOLD = 0.7
EMBEDDING_MODEL = "text-embedding-3-large"


def run(retriever):
    # 1. LOAD the golden set --- the fixed, human-authored truth
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE INJECTED RETRIEVER on each question to fill retrieval_context,
    #    run the generator on what was retrieved, then build one test case per golden.
    test_cases = []
    for g in goldens:
        retrieved = retriever.invoke(g["query"])
        retrieval_context = [doc.page_content for doc in retrieved]
        answer = generate(g["query"], retrieval_context)   # actual_output from the real chunks

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output=answer,
            )
        )

    # 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]

    # 4. EVALUATE --- every metric on every case, batched + parallel, printed report.
    #    hyperparameters travel with the run so the report is tagged with the config.
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": "reranker",          # label for this retriever config
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": 800,
            "chunk_overlap": 100,
            "fetch_k": 10,
            "top_k": 5,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_PATH,
        },
    )
    return summarize_by_metric(result)


if __name__ == "__main__":
    print_summary("retriever", run(RerankingRetriever()))