from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _make_console_handler() -> logging.Handler:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    return h


def setup_logging(cfg: DictConfig) -> dict[str, Any]:
    """Create the experiment dir, configure root logger, init wandb if enabled.

    Returns ``{"logger", "wandb_run", "exp_dir"}``.
    """
    exp_dir = Path(cfg.log_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(_make_console_handler())
    file_handler = logging.FileHandler(str(exp_dir / "experiment.log"))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(file_handler)

    OmegaConf.save(cfg, exp_dir / "config.yaml", resolve=True)

    wandb_run = None
    if bool(cfg.logging.wandb):
        import wandb  # imported lazily so wandb is not required when disabled

        run_name = cfg.logging.get("name", None) or exp_dir.name
        wandb_run = wandb.init(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            group=cfg.logging.group,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(exp_dir),
            tags=list(cfg.logging.tags or []),
        )

    logger.info(f"Experiment dir: {exp_dir}")
    return {"logger": logger, "wandb_run": wandb_run, "exp_dir": exp_dir}


class MetricLogger:
    """Routes one metrics dict to JSONL, console, and (optionally) wandb."""

    def __init__(
        self,
        jsonl_path: Path,
        *,
        console_logger: logging.Logger,
        wandb_run=None,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.console = console_logger
        self.wandb = wandb_run
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def info(self, msg: str) -> None:
        """One-off info line to console + experiment.log (no wandb, no JSONL)."""
        self.console.info(msg)

    def log(self, step: int, metrics: dict[str, Any], summary: str | None = None) -> None:
        """Log a per-step metrics dict.

        - metrics (full, incl. lists/dicts) -> metrics.jsonl.
        - Scalar entries → wandb.log(..., step=step) if wandb is enabled.
        - summary if provided -> console (one line, no wandb).
        """
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
        if self.wandb is not None:
            scalars = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            self.wandb.log(scalars, step=step)
        if summary:
            self.console.info(summary)
