from os.path import join
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from gvcore.utils.logger import logger
from gvcore.utils.distributed import is_main_process


class Checkpointer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dev_mode = cfg.dev_mode

    def save(self, name, **checkpointables):
        if not is_main_process() or self.dev_mode:
            return

        if self.cfg.log.get("keep_last", 0) > 0:
            existing_ckps = os.listdir(self.cfg.log.log_dir)
            existing_ckps = [ckp for ckp in existing_ckps if ckp.endswith(".pth")]
            existing_ckps.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            if len(existing_ckps) >= self.cfg.log.get("keep_last", 0):
                os.remove(join(self.cfg.log.log_dir, existing_ckps[0]))

        data = {}
        for key, obj in checkpointables.items():
            if isinstance(obj, DistributedDataParallel):
                data[key] = obj.module.state_dict()
            elif isinstance(obj, (nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler)):
                data[key] = obj.state_dict()
            else:
                data[key] = obj

        file_path = join(self.cfg.log.log_dir, f"{name}.pth")
        torch.save(data, file_path)
        logger.info(f"Saved checkpoint to {file_path}.")

    def load(self, ckp_path, **checkpointables):
        ckp_dict = torch.load(ckp_path, map_location="cpu")
        except_keys = []
        for key, obj in checkpointables.items():
            if isinstance(obj, DistributedDataParallel):
                load_obj = obj.module
            else:
                load_obj = obj
            if key in ckp_dict:
                if ckp_dict[key] is not None:
                    if isinstance(load_obj, nn.Module):
                        except_param_keys = load_obj.load_state_dict(ckp_dict[key], strict=False)
                        logger.info("Unmatched params:" + str(except_param_keys))
                    elif isinstance(load_obj, (torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler)):
                        load_obj.load_state_dict(ckp_dict[key])
                else:
                    except_keys.append(key)
            else:
                except_keys.append(key)
        except_string = "" if len(except_keys) == 0 else f", except: {', '.join(except_keys)}"
        logger.info(f"Loaded checkpoint from {ckp_path}{except_string}.")

        if "step" in ckp_dict:
            return ckp_dict["step"]
        else:
            return 0
