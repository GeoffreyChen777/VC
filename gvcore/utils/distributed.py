import os
import copy
import torch
import torch.distributed as dist
import functools
import logging
import pickle
import sys
import signal
import subprocess


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def init_distributed(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{cfg.distributed.dist_local_rank}"
    try:
        dist.init_process_group(
            backend="nccl",
            init_method=cfg.distributed.dist_url,
            world_size=cfg.distributed.world_size,
            rank=cfg.distributed.dist_global_rank,
        )
    except Exception as e:
        logger = logging.getLogger()
        logger.error(f"Distributed Initialization Error URL: {cfg.distributed.dist_url}")
        raise e
    synchronize()


def launch_distributed(args):
    if args.dist_url is None:
        raise ValueError("Multi-node distributed training requires a main node url.")
    print(
        f"=> Distributed running on {args.dist_url} with {args.world_size} GPUs."
    )

    processes = []
    os.environ["OMP_NUM_THREADS"] = "1"

    for local_gpu_id in range(0, args.num_gpus):
        cmd = [sys.executable, "run.py"]
        local_args = copy.deepcopy(args)
        if local_args.dist_local_rank == -1:
            local_args.dist_local_rank = local_gpu_id
        if local_args.dist_global_rank == -1:
            local_args.dist_global_rank = local_gpu_id

        for args_name in local_args.__dict__:
            cmd_string = f"--{args_name}={getattr(local_args, args_name)}"

            if args_name == "opt":
                cmd_string = cmd_string[6:]
            elif args_name == "dev":
                if getattr(local_args, args_name):
                    cmd_string = "--dev"
                else:
                    continue
            elif args_name == "modify":
                modifies = getattr(local_args, args_name)
                if modifies is not None:
                    cmd.append("--modify")
                    cmd.extend(modifies)
                continue
            elif args_name == "launch":
                cmd_string = "--launch"
            elif args_name == "slurm":
                continue
            elif getattr(local_args, args_name) is None:
                continue
            
            cmd.append(cmd_string)
        current_env = os.environ.copy()
        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        os.kill(process.pid, signal.SIGINT)

        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def get_local_rank_from_slurm():
    if "SLURM_LOCALID" in os.environ:
        return int(os.environ["SLURM_LOCALID"])
    else:
        raise ValueError("Cannot get local rank from SLURM_LOCALID.")

def get_global_rank_from_slurm():
    if "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])
    else:
        raise ValueError("Cannot get global rank from SLURM_PROCID.")

def get_world_size_from_slurm():
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"])
    else:
        raise ValueError("Cannot get world size from SLURM_NTASKS.")

def get_dist_url_from_slurm():
    try:
        with open("./.slurm.hosts") as reader:
            master_node = reader.readlines()[0].strip()
        with open("./.slurm.port") as reader:
            master_port = reader.readlines()[0].strip()
        return f"tcp://{master_node}:{master_port}"
    except Exception as e:
        print(f"Cannot construct dist_url: {e}")

def is_main_process():
    return get_rank() == 0


def is_distributed():
    return dist.is_initialized()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data.cpu() if isinstance(data, torch.Tensor) else data)
    if len(buffer) > 1024 ** 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert world_size >= 1, "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group=group) == 1:
        return [data]
    rank = dist.get_rank(group=group)

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    # receiving Tensor from all ranks
    if rank == dst:
        max_size = max(size_list)
        tensor_list = [torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list]
        dist.gather(tensor, tensor_list, dst=dst, group=group)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        return data_list
    else:
        dist.gather(tensor, [], dst=dst, group=group)
        return []


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list
