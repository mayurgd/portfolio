"""
Generic HuggingFace Dataset Loader for RAG Evaluation

Converts any HuggingFace dataset into the standard dict format expected
by the evaluation framework:

    {
        "questions":     list[str],   # required
        "answers":       list[str],   # optional – omit if not in dataset
        "contexts":      list[list[str]],  # optional
        "ground_truths": list[str],   # optional
    }

Usage example – FiQA ragas_eval_v3 (pre-built eval split):

    from src.hf_dataset import load_hf_dataset

    data = load_hf_dataset(
        dataset_name="vibrantlabsai/fiqa",
        config_name="ragas_eval_v3",
        split="baseline",
        column_mapping={
            "questions":     "user_input",
            "answers":       "response",
            "contexts":      "retrieved_contexts",
            "ground_truths": "reference",
        },
    )

Usage example – any other QA dataset:

    data = load_hf_dataset(
        dataset_name="some-org/some-dataset",
        split="test",
        column_mapping={
            "questions":     "query",
            "ground_truths": "answer",
        },
        max_samples=50,
    )
"""

from typing import Any

from datasets import load_dataset


def load_hf_dataset(
    dataset_name: str,
    column_mapping: dict[str, str],
    split: str = "test",
    config_name: str | None = None,
    max_samples: int | None = None,
    trust_remote_code: bool = False,
) -> dict[str, list[Any]]:
    """Load a HuggingFace dataset split and return a framework-compatible dict.

    Parameters
    ----------
    dataset_name:
        HuggingFace dataset identifier, e.g. ``"vibrantlabsai/fiqa"``.
    column_mapping:
        Maps framework field names to dataset column names.
        Recognised keys: ``"questions"``, ``"answers"``, ``"contexts"``,
        ``"ground_truths"``.  Any key absent from the mapping is omitted
        from the returned dict.
    split:
        Dataset split to load (``"train"``, ``"test"``, ``"validation"``,
        ``"baseline"``, etc.).
    config_name:
        Dataset configuration / subset name. Pass ``None`` for datasets
        without named configurations.
    max_samples:
        If given, only the first *max_samples* rows are returned.
    trust_remote_code:
        Passed through to ``datasets.load_dataset``.

    Returns
    -------
    dict with a subset of ``questions``, ``answers``, ``contexts``,
    ``ground_truths`` — whichever keys were present in *column_mapping*.
    """
    load_kwargs: dict[str, Any] = {
        "split": split,
        "trust_remote_code": trust_remote_code,
    }
    if config_name is not None:
        load_kwargs["name"] = config_name

    ds = load_dataset(dataset_name, **load_kwargs)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    result: dict[str, list[Any]] = {}
    for field, col in column_mapping.items():
        if col not in ds.column_names:
            continue
        result[field] = _normalise(field, ds[col])

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(field: str, values: list[Any]) -> list[Any]:
    """Coerce raw column values into the type expected per framework field."""
    if field == "ground_truths":
        # Some datasets store ground_truths as list[list[str]]; take first.
        return [
            v[0] if isinstance(v, list) and v else str(v)
            for v in values
        ]
    if field == "contexts":
        # Ensure each entry is a list[str], not a bare string.
        return [
            v if isinstance(v, list) else [str(v)]
            for v in values
        ]
    return list(values)
