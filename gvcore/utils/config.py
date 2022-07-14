import yaml
from easydict import EasyDict
import argparse
import os
from mergedeep import merge, Strategy

from gvcore.utils.distributed import (
    _find_free_port,
    get_dist_url_from_slurm,
    get_global_rank_from_slurm,
    get_local_rank_from_slurm,
)


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

    return EasyDict(base_cfg)


def parse_config(args):
    assert args.config.endswith(".yaml"), f"Only support YAML config file! Wrong config path: {args.config}"

    cfg = parse_yaml(args.config)

    if args.resume is not None:
        assert args.resume.endswith(".pth"), f"Only support resume from .pth file! Wrong weight path: {args.resume}"
        cfg.resume = args.resume
    else:
        cfg.resume = None
    if args.dev or args.opt == "test":
        cfg.dev_mode = True

    cfg.opt = args.opt

    if args.modify is not None:
        modifies = args.modify
        for modify in modifies:
            k, v = modify.split("=")
            if not v.replace(".", "", 1).isdigit() and v != "True" and v != "False":
                exec(f"cfg.{k}='{v}'")
            elif v.replace(".", "", 1).isdigit():
                exec(f"cfg.{k}=float({v})")
            else:
                exec(f"cfg.{k}={v}")

    cfg.distributed.use = args.world_size > 1
    if cfg.distributed.use:
        cfg.distributed.dist_url = args.dist_url
        cfg.distributed.num_gpus = args.num_gpus
        cfg.distributed.world_size = args.world_size
        cfg.distributed.dist_local_rank = args.dist_local_rank
        cfg.distributed.dist_global_rank = args.dist_global_rank

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
    parser.add_argument("--launch", action="store_true", help="To launch operations.")
    parser.add_argument("--dev", action="store_true", help="To use dev mode.")
    parser.add_argument("--config", type=str, default="", help="Path to config file.")
    parser.add_argument("--modify", nargs="+", default=None, help="Modify config file.")
    parser.add_argument(
        "--resume", type=str, default=None, help="Whether to attempt to resume from the checkpoint directory."
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="GPU nums.")
    parser.add_argument("--dist_url", default=None, type=str, help="URL for initilizing distributed running.")
    parser.add_argument("--world_size", default=-1, type=int, help="World size.")
    parser.add_argument("--dist_global_rank", default=-1, type=int, help="Distributed Node Rank.")
    parser.add_argument("--dist_local_rank", default=-1, type=int, help="Distributed Local Rank.")

    parser.add_argument(
        "--slurm",
        default=False,
        action="store_true",
        help="SLURM flag. If set to True, use some environment variable to complete the args.",
    )
    parser.add_argument("--slurm_offset", default=0, type=int, help="Distributed Rank Offset.")
    args = parser.parse_args()

    # For SLURM
    if args.world_size == -1:
        args.world_size = args.num_gpus
    if args.slurm:
        if args.dist_url is None:
            args.dist_url = get_dist_url_from_slurm()
        try:
            if args.dist_global_rank == -1:
                args.dist_global_rank = get_global_rank_from_slurm() + args.slurm_offset
            if args.dist_local_rank == -1:
                args.dist_local_rank = get_local_rank_from_slurm()
        except Exception as e:
            print(f"Cannot launch distribution training: {e}")

        args.num_gpus = 1
    else:
        if args.dist_url is None:
            try:
                master_node = "127.0.0.1"
                master_port = _find_free_port()
                args.dist_url = f"tcp://{master_node}:{master_port}"
            except Exception as e:
                print(f"Cannot construct dist_url: {e}")
    return args
