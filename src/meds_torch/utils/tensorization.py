"""Utility functions for converting tokenized data to nested ragged tensors."""

from pathlib import Path

import polars as pl
from loguru import logger
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict


def convert_to_NRT(parquet_fp: Path) -> JointNestedRaggedTensorDict:
    """Convert a tokenized parquet file into a nested ragged tensor.

    This function reads event sequences from the tokenization stage and converts them
    to JointNestedRaggedTensorDict format for efficient loading by PyTorch dataloaders.

    Args:
        parquet_fp: Path to the tokenized parquet file containing columns:
            - subject_id
            - time_delta_days (list of floats, ragged)
            - code (list of list of ints, doubly ragged)
            - numeric_value (list of list of floats, doubly ragged)
            - modality_idx (optional, list of list of floats, doubly ragged)

    Returns:
        A JointNestedRaggedTensorDict object representing the tokenized dataframe.

    Raises:
        ValueError: If there are no time delta columns or multiple time delta columns.
        FileNotFoundError: If the input parquet file does not exist.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> import polars as pl
        >>>
        >>> # Create test data
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 2],
        ...     "time_delta_days": [[float("nan"), 12.0], [float("nan")]],
        ...     "code": [[[101, 102], [103]], [[201, 202]]],
        ...     "numeric_value": [[[2.0, 3.0], [4.0]], [[6.0, 7.0]]]
        ... })
        >>>
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     fp = Path(tmpdir) / "test.parquet"
        ...     df.write_parquet(fp)
        ...     nrt = convert_to_NRT(fp)
        ...     print(sorted(nrt.tensors.keys()))
        ['dim0/time_delta_days', 'dim1/bounds', 'dim1/code', 'dim1/numeric_value']
    """
    if not parquet_fp.exists():
        raise FileNotFoundError(f"Input file not found: {parquet_fp}")

    # Read the parquet file
    df = pl.read_parquet(parquet_fp, use_pyarrow=True).lazy()

    # Find time delta column
    time_delta_cols = [c for c in df.collect_schema().names() if c.startswith("time_delta_")]

    if len(time_delta_cols) == 0:
        raise ValueError("Expected at least one time delta column, found none")
    elif len(time_delta_cols) > 1:
        raise ValueError(f"Expected exactly one time delta column, found columns: {time_delta_cols}")

    time_delta_col = time_delta_cols[0]

    # Determine which columns to include
    required_cols = [time_delta_col, "code", "numeric_value"]
    available_cols = df.collect_schema().names()

    # Add optional modality_idx if present
    cols_to_select = required_cols.copy()
    if "modality_idx" in available_cols:
        cols_to_select.append("modality_idx")

    # Convert to dict format for JointNestedRaggedTensorDict
    tensors_dict = df.select(cols_to_select).collect().to_dict(as_series=False)

    # Check for empty data
    if all((not v) for v in tensors_dict.values()):
        logger.warning("All columns are empty. Returning an empty tensor dict.")
        return JointNestedRaggedTensorDict({})

    for k, v in tensors_dict.items():
        if not v:
            raise ValueError(f"Column {k} is empty")

    return JointNestedRaggedTensorDict(tensors_dict)
