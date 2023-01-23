from collections import Counter
from copy import deepcopy
from enum import Enum
from functools import partial, reduce
from typing import Mapping, MutableMapping
import yaml
from addict import Dict
import argparse
import os


class Strategy(Enum):
    REPLACE = 0
    ADDITIVE = 1


def _is_recursive_merge(a, b):
    both_mapping = isinstance(a, Mapping) and isinstance(b, Mapping)
    both_counter = isinstance(a, Counter) and isinstance(b, Counter)
    return both_mapping and not both_counter


def _deepmerge(dst, src, strategy):
    for key in src:
        if key in dst:
            if _is_recursive_merge(dst[key], src[key]):
                # If the key for both `dst` and `src` are both Mapping types (e.g. dict), then recurse.
                _deepmerge(dst[key], src[key], strategy)
            elif dst[key] is src[key]:
                # If a key exists in both objects and the values are `same`, the value from the `dst` object will be used.
                pass
            else:
                _handle_merge.get(strategy)(dst, src, key)
        else:
            # If the key exists only in `src`, the value from the `src` object will be used.
            dst[key] = deepcopy(src[key])
    return dst


def merge(destination: MutableMapping, *sources: Mapping, strategy: Strategy = Strategy.REPLACE) -> MutableMapping:
    """
    A deep merge function for 🐍.

    :param destination: The destination mapping.
    :param sources: The source mappings.
    :param strategy: The merge strategy.
    :return:
    """
    return reduce(partial(_deepmerge, strategy=strategy), sources, destination)


def _handle_merge_add(destination, source, key):
    # Values are combined into one long collection.
    if isinstance(destination[key], list) and isinstance(source[key], list):
        # Extend destination if both destination and source are `list` type.
        destination[key] = deepcopy(source[key])
    elif isinstance(destination[key], set) and isinstance(source[key], set):
        # Update destination if both destination and source are `set` type.
        destination[key].update(deepcopy(source[key]))
    elif isinstance(destination[key], tuple) and isinstance(source[key], tuple):
        # Update destination if both destination and source are `tuple` type.
        destination[key] = deepcopy(source[key])
    elif isinstance(destination[key], Counter) and isinstance(source[key], Counter):
        # Update destination if both destination and source are `Counter` type.
        destination[key].update(deepcopy(source[key]))
    else:
        _handle_merge[Strategy.REPLACE](destination, source, key)


def _handle_merge_replace(destination, source, key):
    if isinstance(destination[key], Counter) and isinstance(source[key], Counter):
        # Merge both destination and source `Counter` as if they were a standard dict.
        _deepmerge(destination[key], source[key])
    else:
        # If a key exists in both objects and the values are `different`, the value from the `source` object will be used.
        destination[key] = deepcopy(source[key])


_handle_merge = {Strategy.ADDITIVE: _handle_merge_add, Strategy.REPLACE: _handle_merge_replace}


def parse_yaml(yaml_path):
    with open(yaml_path, "r") as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.Loader)
        if "_base" in cfg:
            yaml_path = os.path.dirname(yaml_path)
            base_path = os.path.join(yaml_path, cfg["_base"])
            with open(base_path, "r") as base_file:
                base_cfg = yaml.load(base_file, Loader=yaml.Loader)
        else:
            base_cfg = {}
        merge(base_cfg, cfg, strategy=Strategy.ADDITIVE)

    return Dict(base_cfg)


def parse_config():
    args = parse_args()
    assert args.config.endswith(".yaml"), f"Only support YAML config file! Wrong config path: {args.config}"

    cfg = parse_yaml(args.config)

    if args.resume is not None:
        assert args.resume.endswith(".pth"), f"Only support resume from .pth file! Wrong weight path: {args.resume}"
        cfg.resume = args.resume
    else:
        cfg.resume = None
    if args.dev or args.opt == "test":
        cfg.dev_mode = True
    cfg.num_gpus = args.num_gpus

    cfg.opt = args.opt

    if args.modify is not None:
        modifies = args.modify
        for modify in modifies:
            k, v = modify.split("=")
            if (
                not v.replace(".", "", 1).isdigit()
                and v != "True"
                and v != "False"
                and not v.startswith("[")
                and not v.endswith("]")
                and not v.startswith("{")
                and not v.endswith("}")
            ):
                exec(f"cfg.{k}='{v}'")
            elif v.replace(".", "", 1).isdigit():
                exec(f"cfg.{k}=float({v})")
            else:
                exec(f"cfg.{k}={v}")
    return cfg


def config_to_string(cfg_dict, level=1):
    all_string = ""
    for k, v in cfg_dict.items():
        if isinstance(v, dict):
            all_string += f"{'  |  '*level}{k}:\n"
            all_string += config_to_string(v, level=level + 1)
        else:
            all_string += f"{'  |  '*level}{k}: {v}\n"
    return all_string


def parse_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("opt", type=str, help="Operation.")
    parser.add_argument("--dev", action="store_true", help="To use dev mode.")
    parser.add_argument("--config", type=str, default="", help="Path to config file.")
    parser.add_argument("--modify", nargs="+", default=None, help="Modify config file.")
    parser.add_argument(
        "--resume", type=str, default=None, help="Whether to attempt to resume from the checkpoint directory."
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="GPUs to use.")

    args = parser.parse_known_args()[0]

    return args
