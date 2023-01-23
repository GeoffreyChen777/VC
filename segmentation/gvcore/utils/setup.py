from gvcore.utils.config import parse_config
from gvcore.utils.logger import logger
from gvcore.utils.distributed import setup_dist_running, is_distributed, get_world_size


def setup():
    # ===================
    # 1. Parse config
    cfg = parse_config()

    # ===================
    # 2. Init distributed running
    setup_dist_running(cfg)

    # ===================
    # 3. Setup logger
    logger.setup_logger(cfg)

    if is_distributed():
        logger.info(f"[!] Distributed Running Initialized: world size = {get_world_size()}")

    return cfg

