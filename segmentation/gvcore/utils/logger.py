import os
import sys
import gc
import torch
import time
import logging
from collections import deque
from datetime import datetime
from os.path import join

import gvcore.utils.distributed as dist_utils

try:
    import wandb

    wandb_installed = True
except:
    wandb_installed = False

try:
    import aim

    aim_installed = True
except:
    aim_installed = False


__all__ = ["logger", "Timer", "Logger", "GenericMetric", "GenericSummary"]


class Timer:
    def __init__(self, window_size=50):
        self.time_window = deque()
        self.window_size = window_size
        self.start_time = None
        self.max_step = 0
        self.is_start = False

    def setup(self, max_step):
        self.max_step = max_step
        self.time_window.clear()
        start_time = time.perf_counter()
        self.time_window.append(start_time)
        self.start_time = start_time
        self.is_start = True

    def record_one_step(self):
        if len(self.time_window) == self.window_size:
            self.time_window.popleft()
        self.time_window.append(time.perf_counter())

    def rest(self, step):
        assert self.is_start, "Please start timer before get time stamp."
        avg_step_time = (self.time_window[-1] - self.time_window[0]) / len(self.time_window)
        total_rest_time = avg_step_time * (self.max_step - step - 1)
        rest_hour, rest_min, rest_sec = self.convert_format(total_rest_time)
        stamp_string = "{}:{}:{}".format(rest_hour, rest_min, rest_sec)
        return stamp_string

    @staticmethod
    def convert_format(sec):
        hour = "{:02}".format(int(sec // 3600))
        minu = "{:02}".format(int((sec % 3600) // 60))
        sec = "{:02}".format(int(sec % 60))
        return hour, minu, sec


class Logger:
    def __init__(self):
        self._logger = None

    def setup_logger(self, cfg):
        if wandb_installed:
            os.environ["WANDB_SILENT"] = "true"

        comment = "_" + cfg.log.comment if cfg.log.comment is not None else ""
        run_time = datetime.now().strftime("%y%m%d%H%M%S")
        logger_name = f"{run_time}{comment}"
        cfg.log.logger_name = logger_name
        cfg.log.run_time = run_time

        logger = logging.getLogger(logger_name)
        logger.propagate = False
        logger.setLevel(logging.DEBUG)

        if dist_utils.is_main_process():
            stream_handler = logging.StreamHandler(stream=sys.stdout)
            stream_handler.setLevel(logging.DEBUG)
            color_start = "\033[32m" if cfg.dev_mode else ""
            color_end = "\033[0m" if cfg.dev_mode else ""
            func_name = "%(funcName)s" if cfg.dev_mode else ""
            stream_formatter = logging.Formatter(
                f"{color_start}[%(asctime)s {func_name}]:{color_end} %(message)s", datefmt="%m/%d %H:%M:%S"
            )
            stream_handler.setFormatter(stream_formatter)
            logger.addHandler(stream_handler)

            if not cfg.dev_mode:
                assert cfg.log.log_dir is not None, "Log path needed!"
                log_dir = os.path.join(cfg.log.log_dir, cfg.log.prefix, logger_name)
                cfg.log.log_dir = log_dir
                os.makedirs(log_dir)
                file_handler = logging.FileHandler(join(log_dir, "log.log"))
                file_handler.setLevel(logging.DEBUG)
                file_formatter = logging.Formatter("[%(asctime)s]: %(message)s", datefmt="%y-%m-%d %H:%M:%S")
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)

        else:
            logger.disabled = True
        self._logger = logger
        self.info = logger.info
        self.error = logger.error
        self.warning = logger.warning
        self.debug = logger.debug


logger = Logger()


class GenericMetric:
    def __init__(self, window_size=20, log_interval=1, distributed=False, printable=True, support_nan=True):
        self.buffer = {}
        self.window_size = window_size
        self.log_interval = log_interval
        self.distributed = distributed
        self.printable = printable
        self.support_nan = support_nan

    def update(self, metrics):
        for key, value in metrics.items():
            assert (
                isinstance(value, (int, float, torch.Tensor)) or value is None
            ), f"GenericMetric only supports int, float, torch.Half/Float/Int/LongTensor or None, but got {type(value)}."

            if key not in self.buffer:
                self.buffer[key] = []

            if value is not None:
                if (
                    torch.isnan(torch.tensor(value) if not torch.is_tensor(value) else value).item()
                    and not self.support_nan
                ):
                    value = self.buffer[key][-1] if len(self.buffer[key]) > 0 else None

            self.buffer[key].append(float(value) if value is not None else None)

    def clear(self):
        del self.buffer
        self.buffer = {}
        gc.collect()

    def _collect(self, window_size=None):
        if len(self.buffer) == 0:
            logger.warning("Nothing in the buffer.")
            return {}
        window_size = float("inf") if window_size is None else window_size
        collected = {}
        for key, values in self.buffer.items():
            num = 0
            collected[key] = []
            for i, value in enumerate(values[::-1]):
                if i >= window_size:
                    break
                if value is None:
                    continue
                num += 1
                collected[key].append(value)
        return collected

    def mean(self):
        collected = self._collect(window_size=self.window_size)

        # Use a new stream so these ops don't wait for DDP or backward
        outputs = {}
        with torch.cuda.stream(torch.cuda.Stream()):
            for key, values in collected.items():
                values = torch.tensor(values).cuda()
                values_num = torch.tensor(values.shape[0]).cuda()
                sum_values = values.sum()
                # Reduce across multiple GPUs.
                if self.distributed:
                    torch.distributed.all_reduce(sum_values)
                    torch.distributed.all_reduce(values_num)
                mean_values = sum_values / values_num
                outputs[key] = float(mean_values)
        return outputs

    def sum(self):
        collected = self._collect(window_size=self.window_size)

        # Use a new stream so these ops don't wait for DDP or backward
        outputs = {}
        with torch.cuda.stream(torch.cuda.Stream()):
            for key, values in collected.items():
                values = torch.tensor(values).cuda()
                sum_values = values.sum()
                # Reduce across multiple GPUs.
                if self.distributed:
                    torch.distributed.all_reduce(sum_values)
                outputs[key] = float(sum_values)
        return outputs

    def last(self):
        collected = self._collect(window_size=1)

        # Use a new stream so these ops don't wait for DDP or backward
        outputs = {}
        with torch.cuda.stream(torch.cuda.Stream()):
            for key, values in collected.items():
                values = torch.tensor(values).cuda()
                # Gather across multiple GPUs.
                if self.distributed:
                    all_last = [torch.zeros(1).cuda() for i in range(self.world_size)]
                    torch.distributed.all_gather(all_last, values)
                    all_last = torch.cat(all_last)
                else:
                    all_last = values
                outputs[key] = all_last
        return outputs


class GenericSummary:
    def __init__(self, cfg, iter_num):
        self.metrics = {}

        self.step = 0
        self.iter_num = iter_num
        self.log_interval = cfg.log.summary_interval

        self.timer = Timer()
        self.timer.setup(self.iter_num)

        # Weight & Bias
        self.use_wandb = (
            dist_utils.is_main_process() and wandb_installed and cfg.log.wandb is not None and not cfg.dev_mode
        )
        if self.use_wandb:
            wandb.init(
                project=cfg.log.wandb,
                config=cfg,
                name=cfg.log.run_time,
                group=cfg.log.prefix,
                resume=cfg.log.wandb_id if "wandb_id" in cfg.log else None,
            )

        # Aim
        self.use_aim = dist_utils.is_main_process() and aim_installed and cfg.log.aim is not None and not cfg.dev_mode
        if self.use_aim:
            self.aim_run = aim.Run(
                repo="aim://132.145.54.161:53800",
                experiment=cfg.log.aim,
                run_hash=cfg.log.aim_hash if "aim_hash" in cfg.log else None,
            )
            self.aim_run["hparams"] = cfg

    def _summary_step(self):
        iter_num_length = len(str(self.iter_num))
        step_length = len(str(self.step))
        step_summary = "[" + " " * (iter_num_length - step_length) + str(self.step) + "/" + str(self.iter_num) + "]"
        return step_summary

    def _collect_metrics(self):
        writable_metrics_collection = {}
        printable_metrics_collection = {}
        for namespace, metric in self.metrics.items():
            if self.step % metric.log_interval == 0:
                mean_v = metric.mean()
                if metric.printable:
                    printable_metrics_collection[namespace] = mean_v
                writable_metrics_collection[namespace] = mean_v
        return writable_metrics_collection, printable_metrics_collection

    def _summary_metrics(self, metrics_collection):
        metrics_summary = []
        for namespace, metric in metrics_collection.items():
            namespace_str = "" if namespace == "default" else namespace + "/"

            metric_string = []
            for k, v in metric.items():
                metric_string.append("{}{}: {:.4f}".format(namespace_str, k, v))

            metric_string = " | ".join(metric_string)
            metrics_summary.append(metric_string)
        return " || ".join(metrics_summary)

    def _write_wandb(self, metric, step, namespace=""):
        if self.use_wandb:
            if namespace == "default" or namespace == "":
                wandb.log(metric, step=step)
            else:
                wandb.log({namespace: metric}, step=step)

    def _write_aim(self, metric, step, namespace=""):
        if self.use_aim:
            for k, v in metric.items():
                if namespace == "default" or namespace == "":
                    self.aim_run.track(v, name=k, step=step)
                else:
                    self.aim_run.track(v, name=k, step=step, context={"namespace": namespace})

    def _write_metrics(self, metrics_collection):
        for namespace, metric in metrics_collection.items():
            # 1. Write to wandb
            if self.use_wandb:
                self._write_wandb(metric, self.step, namespace)

            # 2. Write to aim
            if self.use_aim:
                self._write_aim(metric, self.step, namespace)

    def summary(self):
        self.step += 1
        self.timer.record_one_step()

        rest_time_string = self.timer.rest(self.step)
        step_summary = self._summary_step()

        writable_metrics_collection, printable_metrics_collection = self._collect_metrics()
        metrics_summary = self._summary_metrics(printable_metrics_collection)
        self._write_metrics(writable_metrics_collection)

        if metrics_summary != "":
            logger.info(f"{step_summary} ETA: {rest_time_string} | {metrics_summary}")

    def add_metrics(self, window_size=None, log_interval=None, namespace="default", printable=True, support_nan=True):
        if namespace not in self.metrics:
            self.metrics[namespace] = GenericMetric(
                window_size=self.log_interval if window_size is None else window_size,
                log_interval=self.log_interval if log_interval is None else log_interval,
                distributed=dist_utils.get_world_size() > 1,
                printable=printable,
                support_nan=support_nan,
            )

    def update(self, metrics, namespace="default", support_nan=True):
        if namespace not in self.metrics:
            self.add_metrics(namespace=namespace, printable=True, support_nan=support_nan)
        self.metrics[namespace].update(metrics)
