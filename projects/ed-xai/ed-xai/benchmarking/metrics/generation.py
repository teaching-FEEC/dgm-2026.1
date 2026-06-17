from typing import List

from .base import MetricFn

from benchmarking.benchmarker import _extract_assistant_text


def _strip_opener(text: str) -> str:
    """Strip the first sentence ('This is a fake/real image.') from the text."""
    idx = text.find(". ")
    return text[idx + 2:].strip() if idx != -1 else text


def _rouge_l(responses: List[str], references: List[str], **kwargs) -> float:
    try:
        from rouge_score import rouge_scorer as rs_module
    except ImportError as e:
        raise ImportError("rouge-score is required: pip install rouge-score") from e

    scorer = rs_module.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for resp, ref in zip(responses, references):
        if ref and resp:
            score = scorer.score(ref, resp)
            scores.append(score["rougeL"].fmeasure)
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 4)


def _css(responses: List[str], references: List[str], **kwargs) -> float:
    """Contextual Semantic Similarity via BERTScore F1, computed on the
    explanation part of the response (second sentence onward)."""
    try:
        from bert_score import score as bert_score_fn
    except ImportError as e:
        raise ImportError("bert-score is required: pip install bert-score") from e

    cands, refs = [], []
    for resp, ref in zip(responses, references):
        if ref and resp:
            explanation = _strip_opener(_extract_assistant_text(resp))
            cands.append(explanation if explanation else resp)
            refs.append(ref)

    if not cands:
        return 0.0

    _, _, f1 = bert_score_fn(cands, refs, lang="en", verbose=False)
    return round(float(f1.mean()), 4)


rouge_l = MetricFn(name="Rouge-L", fn=_rouge_l)
css = MetricFn(name="CSS", fn=_css)
