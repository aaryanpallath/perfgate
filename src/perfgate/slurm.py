"""Emit a Slurm job-array script that runs a sweep with one array task per case.

Each array task runs every repeat of its case and writes its own JSONL shard, so tasks
never contend for one output file; `perfgate merge` joins the shards afterwards.
"""

from __future__ import annotations

import shlex

from .config import Sweep


def sbatch_script(
    sweep: Sweep,
    config_path: str,
    *,
    partition: str | None = None,
    gpus_per_task: int = 1,
    time_limit: str = "00:30:00",
    out_dir: str = "results/shards",
    container: str | None = None,
) -> str:
    n = len(sweep.cases)
    if n == 0:
        raise ValueError("sweep has no cases")
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=perfgate-{sweep.name}",
        f"#SBATCH --array=0-{n - 1}",
        "#SBATCH --ntasks=1",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --output={out_dir}/%A_%a.log",
        # Whole node: other jobs on the same GPU or memory bus would contaminate timings.
        "#SBATCH --exclusive",
    ]
    if gpus_per_task:
        lines.append(f"#SBATCH --gpus-per-task={gpus_per_task}")
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    cmd = (
        f"perfgate sweep {shlex.quote(config_path)} --case-index \"$SLURM_ARRAY_TASK_ID\" "
        f"--out {shlex.quote(out_dir)}/\"$SLURM_ARRAY_TASK_ID\".jsonl"
    )
    if container:
        cmd = f"singularity exec --nv {shlex.quote(container)} {cmd}"
    lines += ["", "set -euo pipefail", f"mkdir -p {shlex.quote(out_dir)}", cmd, ""]
    return "\n".join(lines)
