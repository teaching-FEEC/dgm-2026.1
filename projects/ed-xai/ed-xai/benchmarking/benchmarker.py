import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd


_FAKE_KW = frozenset(
    {"fake", "synthetic", "manipulated", "generated", "artificial",
     "forged", "deepfake", "fabricated", "inauthentic", "altered"}
)
_REAL_KW = frozenset(
    {"real", "authentic", "genuine", "original", "unmodified", "natural"}
)


def _extract_assistant_text(response: str) -> str:
    """Return only the assistant's reply, stripping the USER/ASSISTANT prompt wrapper."""
    marker = "ASSISTANT:"
    idx = response.find(marker)
    return response[idx + len(marker):].strip() if idx != -1 else response


def _response_to_pred_and_score(response: str):
    """Derive binary prediction from the first sentence of the model response.

    Matches the FakeVLM eval_vllm.py logic: check if 'real' or 'fake' appears
    in the first sentence (split on '.'), with 'real' taking priority.
    Returns (y_pred, y_score): y_pred is 0=fake/1=real, y_score is 1.0/0.0.
    Returns (-1, 0.0) when neither keyword is found, ensuring unrecognized
    responses are counted as errors regardless of ground truth.
    """
    first_sentence = _extract_assistant_text(response).split('.')[0].lower()
    if 'real' in first_sentence:
        return 1, 1.0
    elif 'fake' in first_sentence:
        return 0, 0.0
    return -1, 0.0


class Benchmarker:
    """
    Cross-model, cross-dataset evaluation framework using pre-computed inference results.

    Usage:

        benchmarker = Benchmarker(
            models=[("FakeVLM-Extended", "FakeVLM_extended/output/test/results.json")],
            datasets=[("FakeClue-Test", "../data/external/FakeClue/data_json/test_frequency.json")],
            metrics=[accuracy, f1, css, rouge_l],
            output_dir="./benchmark_output",
        )
        benchmarker.run().export("benchmark")

    models: list of (name, results_json_path)
        results_json_path — path to eval.py output JSON, a list of dicts:
            "image":               str  — image path (join key with dataset)
            "response":            str  — model text response
            "ground_truth_label":  int  — 0=fake, 1=real (informational only;
                                          ground truth is always taken from the dataset JSON)

    datasets: list of (name, dataset_json_path)
        dataset_json_path — path to FakeClue JSON (e.g. test_frequency.json), a list of dicts:
            "image":         str   — image path (join key)
            "label":         int   — 0=fake, 1=real
            "conversations": list  — [{"from": "human", ...}, {"from": "gpt", "value": <reference>}]

    Predicted labels and confidence scores are derived from the model response text via
    keyword matching (fake/real vocabulary). Ground-truth labels and reference responses
    always come from the dataset JSON.
    """

    def __init__(
        self,
        models: Optional[List] = None,
        datasets: Optional[List] = None,
        metrics: Optional[List] = None,
        eval_pairs: Optional[List] = None,
        output_dir: Union[str, Path] = "./benchmark_results",
    ):
        self._models: List[Dict] = []
        self._datasets: List[Dict] = []
        self._metrics: List = []
        self._eval_pairs: List[tuple] = []
        self._results: Optional[pd.DataFrame] = None
        self.output_dir = Path(output_dir)

        for item in models or []:
            self.add_model(*item)
        for item in datasets or []:
            self.add_dataset(*item)
        for metric in metrics or []:
            self.add_metric(metric)
        for pair in eval_pairs or []:
            self.add_eval_pair(*pair)

    # ------------------------------------------------------------------
    # Builder methods
    # ------------------------------------------------------------------

    def add_model(self, name: str, results_json_path: Union[str, Path]) -> "Benchmarker":
        self._models.append({"name": name, "path": Path(results_json_path)})
        return self

    def add_dataset(self, name: str, dataset_json_path: Union[str, Path]) -> "Benchmarker":
        self._datasets.append({"name": name, "path": Path(dataset_json_path)})
        return self

    def add_metric(self, metric) -> "Benchmarker":
        self._metrics.append(metric)
        return self

    def add_eval_pair(self, model_name: str, dataset_name: str) -> "Benchmarker":
        self._eval_pairs.append((model_name, dataset_name))
        return self

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> "Benchmarker":
        """Join model results with dataset ground truth and compute metrics."""
        if not self._models:
            raise ValueError("No models registered. Use add_model() first.")
        if not self._datasets:
            raise ValueError("No datasets registered. Use add_dataset() first.")
        if not self._metrics:
            raise ValueError("No metrics registered. Use add_metric() first.")

        models_by_name = {m["name"]: m for m in self._models}
        datasets_by_name = {d["name"]: d for d in self._datasets}

        if self._eval_pairs:
            for model_name, dataset_name in self._eval_pairs:
                if model_name not in models_by_name:
                    raise ValueError(f"Unknown model '{model_name}'. Registered: {list(models_by_name)}")
                if dataset_name not in datasets_by_name:
                    raise ValueError(f"Unknown dataset '{dataset_name}'. Registered: {list(datasets_by_name)}")
            pairs = [(models_by_name[m], datasets_by_name[d]) for m, d in self._eval_pairs]
        else:
            pairs = [(m, d) for m in self._models for d in self._datasets]

        records = []
        for model, dataset in pairs:
            with open(model["path"]) as f:
                raw_results = json.load(f)
            results_by_image = {r["image"]: r for r in raw_results}

            with open(dataset["path"]) as f:
                entries = json.load(f)

            y_true, y_pred, y_score, responses, references = [], [], [], [], []
            skipped = 0

            for entry in entries:
                key = entry["image"]
                if key not in results_by_image:
                    skipped += 1
                    continue

                result = results_by_image[key]

                label = entry["label"]
                if isinstance(label, str):
                    label = 0 if label.lower() == "fake" else 1
                y_true.append(int(label))

                convs = entry.get("conversations", [])
                references.append(convs[1]["value"] if len(convs) > 1 else "")

                response = result.get("response", "")
                responses.append(response)

                pred, score = _response_to_pred_and_score(response)
                y_pred.append(pred)
                y_score.append(score)

            tag = f"[{model['name']} × {dataset['name']}]"
            matched = len(y_true)
            print(f"{tag} matched {matched}/{matched + skipped} entries"
                  + (f" ({skipped} skipped)" if skipped else ""))

            row: Dict = {"Model": model["name"], "Dataset": dataset["name"]}
            for metric in self._metrics:
                val = metric(
                    y_true=y_true,
                    y_pred=y_pred,
                    y_score=y_score,
                    responses=responses,
                    references=references,
                )
                if isinstance(val, dict):
                    row.update(val)
                else:
                    row[metric.name] = val
            records.append(row)

        self._results = pd.DataFrame(records).set_index(["Model", "Dataset"])
        return self

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @property
    def results(self) -> pd.DataFrame:
        if self._results is None:
            raise RuntimeError("Call run() before accessing results.")
        return self._results

    def export_xlsx(self, path: Optional[Union[str, Path]] = None) -> "Benchmarker":
        from .export import export_xlsx
        out = Path(path) if path else self.output_dir / "benchmark.xlsx"
        out.parent.mkdir(parents=True, exist_ok=True)
        export_xlsx(self.results, out)
        print(f"Saved: {out}")
        return self

    def export_png(self, path: Optional[Union[str, Path]] = None) -> "Benchmarker":
        from .export import export_png
        out = Path(path) if path else self.output_dir / "benchmark.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        export_png(self.results, out)
        print(f"Saved: {out}")
        return self

    def export(self, stem: Optional[Union[str, Path]] = None) -> "Benchmarker":
        """Export both .xlsx and .png. stem may be a path without extension."""
        if stem is None:
            return self.export_xlsx().export_png()
        stem = Path(stem)
        return self.export_xlsx(stem.with_suffix(".xlsx")).export_png(stem.with_suffix(".png"))
