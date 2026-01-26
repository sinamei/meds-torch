"""Tensorization stage for MEDS-transforms pipeline.

This stage converts tokenized event sequences into nested ragged tensor (NRT) format
for efficient loading by PyTorch dataloaders. It reads event_seqs/*.parquet files
and writes data/*.nrt files using the JointNestedRaggedTensorDict format.
"""

import logging
from functools import partial

from MEDS_transforms.mapreduce.shard_iteration import shard_iterator
from MEDS_transforms.stages import Stage
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict
from omegaconf import DictConfig, OmegaConf

from meds_torch.utils.tensorization import convert_to_NRT

logger = logging.getLogger(__name__)


def main(cfg: DictConfig):
    """Convert tokenized event sequences to nested ragged tensor format.

    This stage processes event_seqs/*.parquet files and creates data/*.nrt files
    that can be efficiently loaded by the PyTorch dataloader.

    The dataloader expects files at: {data_dir}/data/{split}/{shard}.nrt
    For example: tokenization/data/train/0.nrt

    Args:
        cfg: Hydra configuration containing:
            - stage_cfg.data_input_dir: Directory containing tokenization/event_seqs/
            - stage_cfg.output_dir: Should be set to data/ directory (e.g., tokenization/data)
    """
    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )

    if train_only := cfg.stage_cfg.get("train_only", False):
        raise ValueError(f"train_only={train_only} is not supported for this stage.")

    # Use shard_iterator with in_prefix to only process event_seqs and out_suffix for .nrt
    shard_iterator_fn = partial(shard_iterator, in_prefix="event_seqs/", out_suffix=".nrt")
    shards_single_output, _ = shard_iterator_fn(cfg)

    for in_fp, out_fp in shards_single_output:
        # Skip if already exists and not overwriting
        if out_fp.exists() and not cfg.do_overwrite:
            logger.info(f"Skipping {in_fp} as {out_fp} already exists")
            continue

        logger.info(f"Converting {str(in_fp.resolve())} to NRT at {str(out_fp.resolve())}")

        # Convert to NRT and save
        try:
            nrt = convert_to_NRT(in_fp)
            out_fp.parent.mkdir(parents=True, exist_ok=True)
            nrt.save(out_fp)
            logger.info(f"Successfully wrote {out_fp}")
        except Exception as e:
            logger.error(f"Failed to process {in_fp}: {e}")
            raise

    logger.info(f"Done with {cfg.stage}")


# Register the stage with MEDS-transforms
# This is a data stage (processes data shards, not metadata)
stage = Stage.register(main_fn=main, is_metadata=False)
