import argparse
import numbers
import os
import subprocess
import atexit
from time import sleep
from typing import AnyStr
import sys
from addict import Dict
import torch
from .distributed import find_free_port


class Launcher:
    def __init__(self):
        self.args = parse_args()
        self.mode = "local"
        self.cluster_name = os.popen("hostname").read()

    def launch(self):
        if self.mode == "slurm":
            self._submit_to_slurm()
        elif self.mode == "local":
            self._launch()
        else:
            raise ValueError(f"Unknown launcher mode: {self.mode}")

    def _submit_to_slurm(self):
        # 1. Get GPU Nums
        if self.args[0].num_gpus is None:
            print("[!] No number of GPUs specified. Using 1 GPU.")
            num_gpus = 1
        else:
            num_gpus = self.args[0].num_gpus
        # 2. Get Job Name
        if self.args[0].config is not None:
            splited_cfg_path = self.args[0].config.split("/")
            job_name = "/".join(splited_cfg_path[-min(len(splited_cfg_path), 2) :])
        else:
            job_name = "gv.run"
        # 3. Get Command String
        if self.args[0].general:
            cmd = ["--general"]
        else:
            cmd = []

        if self.args[0].num_gpus is not None and not self.args[0].general:
            cmd.extend(["--num-gpus", str(num_gpus)])

        cmd.extend(self.args[1])
        if self.args[0].config is not None and not self.args[0].general:
            cmd.extend(["--config", self.args[0].config])
        cmd_str = "gvrun --slurm " + " ".join(cmd)
        # 5. Submit to Cluster
        if "avon" in self.cluster_name:
            self._submit_to_avon(job_name, num_gpus, cmd_str)
        elif "wmg" in self.cluster_name:
            self._sumbit_to_wmg(job_name, num_gpus, cmd_str)
        elif "aber" in self.cluster_name:
            self._submit_to_aber(job_name, num_gpus, cmd_str)

    def _submit_to_avon(self, job_name: AnyStr, gpus: numbers.Number, cmd_str: AnyStr):
        world_size = gpus
        node_list = Dict({"gpu" + f"{k}".zfill(3): Dict({"used": 0, "used_by_me": False}) for k in range(1, 17)})

        # 1. Get Node used by me
        jobs = os.popen("squeue -u wmrkwh").read().split("\n")[1:]
        for job in jobs:
            if job != "":
                node_str_list1 = job.split(" ")[-1].split(",")
                for node_str in node_str_list1:
                    node_str = node_str.replace("gpu", "").replace("[", "").replace("]", "")

                    if "-" in node_str:
                        node_start, node_end = node_str.split("-")
                        for i in range(int(node_start), int(node_end) + 1):
                            node_list["gpu" + f"{i}".zfill(3)]["used_by_me"] = True
                    else:
                        node_list["gpu" + node_str.zfill(3)]["used_by_me"] = True

        # 2. Get Free Resource
        for i in range(1, 17):
            node_id = f"{i}".zfill(3)
            cmd = f"scontrol show node gpu{node_id} | grep AllocTRES"
            alloc_gpu = os.popen(cmd).read()
            alloc_gpu = alloc_gpu.split(",")[-1].strip()
            try:
                k, v = alloc_gpu.split("=")
                if k == "gres/gpu":
                    used_gpu_num = int(v)
                elif k == "AllocTRES":
                    used_gpu_num = 0
                else:
                    used_gpu_num = 3
                node_list[f"gpu{node_id}"].used = used_gpu_num
            except:
                continue

        total_free_gpus = 0
        for _, node_info in node_list.items():
            total_free_gpus += 3 - node_info.used

        if total_free_gpus < gpus:
            specific_node_str = ""
            specific_node_str_additional = ""
        else:
            weighted_node_list = Dict()
            for node_name, node_info in node_list.items():
                weighted_node_list[node_name] = node_info
                weighted_node_list[node_name].weight = (
                    (3 - node_info.used) + 100 * (1 if node_info.used_by_me else 0)
                ) * (1 if 3 - node_info.used > 0 else 0)

            sorted_node_list = sorted(weighted_node_list.items(), key=lambda x: x[1].weight, reverse=True)
            specific_node_list = []
            specific_node_gpus = 0
            for node_name, node_info in sorted_node_list:
                if specific_node_gpus >= gpus:
                    break
                specific_node_list.append(node_name)
                specific_node_gpus += 3 - node_info.used

            if len(specific_node_list) <= 4:
                specific_node_str = "#SBATCH -w " + ",".join(specific_node_list)
                specific_node_str_additional = ""
            else:
                specific_node_str = "#SBATCH -w " + ",".join(specific_node_list[:4])
                specific_node_str_additional = "#SBATCH -w " + ",".join(specific_node_list[4:])

                gpus_additional = gpus - sum([3 - node_list[node_name].used for node_name in specific_node_list[:4]])
                gpus = sum([3 - node_list[node_name].used for node_name in specific_node_list[:4]])

        # 3. Submit Job
        if specific_node_str_additional == "":
            sbatch_template = f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -o ./slurmlog/%j.out
#SBATCH --partition=gpu
#SBATCH --ntasks={gpus}
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2-00:00:00
{specific_node_str}

export MASTER_ADDR=`hostname`
export MASTER_PORT=`comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
export WORLD_SIZE={world_size}

srun {cmd_str}
"""
            with open("./.avon_launch.sh", "w") as f:
                f.write(sbatch_template)

            p = subprocess.Popen(
                ["sbatch", "./.avon_launch.sh"], stdout=subprocess.PIPE
            )
            print(p.stdout.read().decode("UTF-8"))
            os.remove("./.avon_launch.sh")

        else:
            sbatch_template = f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -o ./slurmlog/%j.out
#SBATCH --partition=gpu
#SBATCH --ntasks={gpus}
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2-00:00:00
{specific_node_str}


export MASTER_ADDR=`hostname`
export MASTER_PORT=`comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
export WORLD_SIZE={world_size}

echo $MASTER_ADDR > .slurm.hosts
echo $MASTER_PORT > .slurm.port

srun {cmd_str}
"""
            with open("./.avon_launch.sh", "w") as f:
                f.write(sbatch_template)

            p = subprocess.Popen(
                ["sbatch", "./.avon_launch.sh"], stdout=subprocess.PIPE
            )
            print(p.stdout.read().decode("UTF-8"))
            os.remove("./.avon_launch.sh")
            sleep(5)

            sbatch_template = f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -o ./slurmlog/%j.out
#SBATCH --partition=gpu
#SBATCH --ntasks={gpus_additional}
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2-00:00:00
{specific_node_str_additional}

export MASTER_ADDR=$(< .slurm.hosts)
export MASTER_PORT=$(< .slurm.port)
export WORLD_SIZE={world_size}
export RANK_OFFSET={gpus}

srun {cmd_str}
"""
            with open("./.avon_launch.sh", "w") as f:
                f.write(sbatch_template)

            p = subprocess.Popen(
                ["sbatch", "./.avon_launch.sh"], stdout=subprocess.PIPE
            )
            print(p.stdout.read().decode("UTF-8"))
            os.remove("./.avon_launch.sh")

    def _sumbit_to_wmg(self, job_name: AnyStr, gpus: numbers.Number, cmd_str: AnyStr):
        if not os.path.exists("./slurmlog"):
            os.makedirs("./slurmlog")

        sbatch_template = f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -o ./slurmlog/%j.out
#SBATCH -p xlong
#SBATCH --ntasks=1
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={gpus*8}
#SBATCH --mem-per-gpu=32G

# export MASTER_ADDR='127.0.0.1'
# export MASTER_PORT=`comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
# export WORLD_SIZE={gpus}

if [[ $(hostname) != "hpc-gpu-03.wmgds.wmg.warwick.ac.uk" ]]; then
echo "Set Proxy"
export http_proxy=http://wmg-squid.wmgds.wmg.warwick.ac.uk:3128
export https_proxy=http://wmg-squid.wmgds.wmg.warwick.ac.uk:3128
else
export http_proxy=''
export https_proxy=''
fi

srun {cmd_str.replace('--slurm', '')}
"""
        with open("./.wmg_launch.sh", "w") as f:
            f.write(sbatch_template)

        p = subprocess.Popen(["sbatch", "./.wmg_launch.sh"], stdout=subprocess.PIPE)
        print(p.stdout.read().decode("UTF-8"))
        os.remove("./.wmg_launch.sh")


    def _submit_to_aber(self, job_name: AnyStr, gpus: numbers.Number, cmd_str: AnyStr):
        if not os.path.exists("./slurmlog"):
            os.makedirs("./slurmlog")

        sbatch_template = f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -o ./slurmlog/%j.out
#SBATCH -p gpu
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={gpus*8}

srun {cmd_str.replace('--slurm', '')}
"""
        with open("./.aber_launch.sh", "w") as f:
            f.write(sbatch_template)

        p = subprocess.Popen(["sbatch", "./.aber_launch.sh"], stdout=subprocess.PIPE)
        print(p.stdout.read().decode("UTF-8"))
        os.remove("./.aber_launch.sh")

    def _launch(self):
        current_env = os.environ.copy()

        if 'aber' in self.cluster_name:
            current_env['NCCL_P2P_DISABLE']="1"
            current_env['NCCL_IB_DISABLE']="1"

        cmd = []
        cmd.extend(self.args[1])

        if self.args[0].config is not None:
            cmd.extend(["--config", self.args[0].config])

        if "CUDA_VISIBLE_DEVICES" not in current_env:
            print("[!] GPU unavailable. Using CPU...")
            process = subprocess.Popen(cmd, env=current_env)
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)
            return

        if "MASTER_ADDR" not in current_env:
            current_env["MASTER_ADDR"] = "127.0.0.1"

        if "MASTER_PORT" not in current_env:
            current_env["MASTER_PORT"] = str(find_free_port())

        if self.args[0].num_gpus is not None:
            current_env["WORLD_SIZE"] = str(self.args[0].num_gpus)
        if "WORLD_SIZE" not in current_env:
            gpu_ids = current_env["CUDA_VISIBLE_DEVICES"].split(",")
            current_env["WORLD_SIZE"] = str(len(gpu_ids))

        if "OMP_NUM_THREADS" not in current_env and int(current_env["WORLD_SIZE"]) > 1:
            current_env["OMP_NUM_THREADS"] = str(1)

        if self.args[0].slurm:
            # Job is lanched by slurm
            current_env["CUDA_VISIBLE_DEVICES"] = current_env["SLURM_LOCALID"]
            current_env["LOCAL_RANK"] = "0"
            current_env["RANK"] = current_env["SLURM_PROCID"]
            if "RANK_OFFSET" in current_env:
                current_env["RANK"] = str(int(current_env["RANK"]) + int(current_env["RANK_OFFSET"]))

            if not self.args[0].general and cmd[0].endswith(".py"):
                cmd.insert(0, f"--master_port={current_env['MASTER_PORT']}")
                cmd.insert(0, f"--master_addr={current_env['MASTER_ADDR']}")
                cmd.insert(0, f"--node_rank={current_env['RANK']}")
                cmd.insert(0, f"--nnodes={current_env['WORLD_SIZE']}")
                cmd.insert(0, "--nproc_per_node=1")
                if torch.__version__ >= "1.10":
                    cmd.insert(0, "torchrun")
                else:
                    cmd.insert(0, f"--use_env")
                    cmd.insert(0, "torch.distributed.launch")
                    cmd.insert(0, "-m")
                    cmd.insert(0, "python")

            process = subprocess.Popen(cmd, env=current_env)
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)
        else:
            # Job is lanched on local
            gpu_ids = current_env["CUDA_VISIBLE_DEVICES"].split(",")
            num_gpus = self.args[0].num_gpus or len(gpu_ids)

            if not self.args[0].general and cmd[0].endswith(".py"):
                cmd.insert(0, f"--nproc_per_node={num_gpus}")
                cmd.insert(0, f"--master_port={current_env['MASTER_PORT']}")
                cmd.insert(0, f"--master_addr={current_env['MASTER_ADDR']}")
                if torch.__version__ >= "1.10":
                    cmd.insert(0, "torchrun")
                else:
                    cmd.insert(0, "--use_env")
                    cmd.insert(0, "torch.distributed.launch")
                    cmd.insert(0, "-m")
                    cmd.insert(0, "python")

            assert (
                len(gpu_ids) >= num_gpus
            ), f"Number of GPUs in the config ({num_gpus}) is greater than the number of GPUs in the environment ({len(gpu_ids)})."

            processes = []
            processes.append(subprocess.Popen(cmd, env=current_env))
            atexit.register(lambda processes: [p.kill() for p in processes], processes)

            for process in processes:
                process.wait()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="Path to config file.")
    parser.add_argument("--num-gpus", type=int, required=False, help="Number of GPUs to use.")
    parser.add_argument("--slurm", action="store_true", default=False, help="If set, job is submitted by slurm.")
    parser.add_argument(
        "--general", action="store_true", default=False, help="If set, job is launched in general mode."
    )

    args = parser.parse_known_args()
    return args


def gvrun():
    launcher = Launcher()
    launcher.mode = "local"
    launcher.launch()


def gvsubmit():
    if not os.path.exists("./slurmlog"):
        os.mkdir("./slurmlog")
    launcher = Launcher()
    launcher.mode = "slurm"
    launcher.launch()
