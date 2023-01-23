import argparse
import os
import subprocess
import time
import wandb


class Sweeper:
    def __init__(self, sweep_id: str, command: str, sweep_count: int = 9999, max_queue: int = 3):
        self.sweep_id = sweep_id
        self.command = command
        self.sweep_count = sweep_count
        self.max_queue = max_queue

    def get_waiting_job_num(self):
        cluster_name = os.popen("hostname").read()

        if "avon" in cluster_name:
            waiting_job_num = self._get_waiting_job_num_avon()
        elif "wmg" in cluster_name:
            waiting_job_num = self._get_waiting_job_num_wmg()

        return waiting_job_num

    def _get_waiting_job_num_avon(self):
        waiting_job_num = 0
        cmd = "squeue -u wmrkwh -h"
        job_list = os.popen(cmd).read().split("\n")
        for job in job_list:
            if job != "":
                stat = job.split()[4]
                if stat != "R":
                    waiting_job_num += 1

        return waiting_job_num

    def _get_waiting_job_num_wmg(self):
        waiting_job_num = 0
        cmd = "squeue -u chen_c -h"
        job_list = os.popen(cmd).read().split("\n")
        for job in job_list:
            if job != "":
                stat = job.split()[4]
                if stat != "R":
                    waiting_job_num += 1

        return waiting_job_num

    def start_agent(self):
        run = wandb.init()

        agent_confg = ""
        for k, v in wandb.config.items():
            if v == "true":
                v = "True"
            elif v == "false":
                v = "False"
            elif isinstance(v, dict):
                v = f'"{v}"'
            v = str(v).replace(" ", "")
            agent_confg += f"{k}={v} "

        agent_confg += f"log.wandb_id={run.id}"
        command = self.command + f" --modify {agent_confg}"
        command_list = ["gvsubmit"]
        command_list.extend(command.split(" "))
        subprocess.run(command_list)

    def sweep(self):
        for _ in range(self.sweep_count):
            waiting_job_num = self.get_waiting_job_num()
            if waiting_job_num < self.max_queue:
                print(f"[*] {waiting_job_num} jobs are waiting in queue. Starting sweep agent.")
                try:
                    wandb.agent(self.sweep_id, function=self.start_agent, count=1)
                except Exception as e:
                    print(f"[*] Sweep stopped.")
                    exit()
            else:
                print(f"[!] {waiting_job_num} jobs are waiting in queue.")
            time.sleep(30)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-id", type=str, required=True, help="Wandb sweep ID.")
    parser.add_argument("--sweep-count", type=int, default=9999, required=False)
    parser.add_argument("--max-queue", type=int, default=3, required=False)
    parser.add_argument("--num-gpus", type=int, required=True, help="Number of GPUs to use.")

    args = parser.parse_known_args()
    return args


def gvsweep():
    os.environ["WANDB_SILENT"] = "true"

    args = parse_args()
    command = " ".join(args[1])
    command += f" --num-gpus {args[0].num_gpus}"
    sweeper = Sweeper(args[0].sweep_id, command, args[0].sweep_count, args[0].max_queue)

    sweeper.sweep()
