from gvcore.utils.config import parse_config, parse_args
import gvcore.utils.distributed as dist_utils
from gvcore.utils.logger import logger
from gvcore.operators import OPERATOR_REGISTRY

# ---------------------------------------------------------------------------- #
# Register

import dataset as _
import evaluator as _
import operators as _

# ---------------------------------------------------------------------------- #


def launch(cfg):
    logger.setup_logger(cfg, dist_utils.is_main_process())

    if cfg.opt == "train":
        operator = OPERATOR_REGISTRY[cfg.operator](cfg)
        operator.train()
    elif cfg.opt == "test":
        operator = OPERATOR_REGISTRY[cfg.operator](cfg)
        operator.test()


if __name__ == "__main__":
    args = parse_args()
    cfg = parse_config(args)
    if not args.launch and cfg.distributed.use:
        dist_utils.launch_distributed(args)
    else:
        if cfg.distributed.use:
            dist_utils.init_distributed(cfg)
        launch(cfg)
