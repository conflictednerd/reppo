import os
import random

import hydra
import numpy as np
import torch
from hydra.utils import get_class
from omegaconf import DictConfig

from src.utils import MetricLogger, setup_logging


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    info = setup_logging(cfg)
    exp_dir = info["exp_dir"]
    logger = MetricLogger(
        jsonl_path=exp_dir / "metrics.jsonl",
        console_logger=info["logger"],
        wandb_run=info["wandb_run"],
    )
    device = torch.device("cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")
    set_seed(cfg.seed)
    cls = get_class(cfg.algorithm._target_)
    algo = cls(
        cfg.algorithm,
        cfg.env,
        exp_dir=exp_dir,
        device=device,
        seed=cfg.seed,
        logger=logger,
    )
    try:
        algo.train()
        algo.save("final")
    finally:
        algo.close()


if __name__ == "__main__":
    main()
