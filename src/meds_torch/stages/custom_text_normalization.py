"""Custom text normalization stage for MEDS-transforms pipeline.

This stage normalizes MEDS data including text_value columns, converting codes
to vocab indices and normalizing numeric values using z-score normalization.
"""

from collections.abc import Callable

import polars as pl
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig

from meds_torch.utils.custom_text_normalization import normalize


@Stage.register
def custom_text_normalization(
    stage_cfg: DictConfig, code_metadata: pl.LazyFrame, code_modifiers: list[str] | None = None
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Normalizes MEDS data with text values using z-score normalization.

    This is a wrapper around the custom_text_normalization utility that integrates
    with MEDS-transforms v0.6.0. It handles both numeric value normalization and
    code to vocab_index conversion for datasets containing text_value columns.

    Args:
        stage_cfg: Configuration for the custom_text_normalization stage.
        code_metadata: Metadata about codes including vocab indices and normalization stats.
        code_modifiers: Optional list of code modifier columns.

    Returns:
        A function that normalizes a MEDS dataframe with text values.
    """

    def transform_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        # Convert LazyFrame to DataFrame for normalize function
        code_metadata_df = code_metadata.collect() if isinstance(code_metadata, pl.LazyFrame) else code_metadata
        return normalize(df, code_metadata_df, code_modifiers)

    return transform_fn
