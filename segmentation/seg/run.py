from gvcore.utils.setup import setup
from gvcore.operators import OPERATOR_REGISTRY

# ---------------------------------------------------------------------------- #
# Register

import dataset as _
import evaluator as _
import operators as _

# ---------------------------------------------------------------------------- #


def main():
    cfg = setup()
    if cfg.opt == "train":
        operator = OPERATOR_REGISTRY[cfg.operator](cfg)
        operator.train()
    elif cfg.opt == "test":
        operator = OPERATOR_REGISTRY[cfg.operator](cfg)
        operator.test()


if __name__ == "__main__":
    main()
