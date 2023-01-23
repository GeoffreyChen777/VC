import torch
import os.path as osp
import json
from tqdm import tqdm

cache_path = "./cache/"

coco_full_set = torch.load(osp.join(cache_path, "coco17_train.dict"))

l_ratio = "10"
ul_ratio = "90"

str_map = {
    "005": "0.5",
    "01": "1.0",
    "02": "2.0",
    "05": "5.0",
    "10": "10.0",
}

for seed in ["1", "2", "3", "4", "5"]:
    l_idx = torch.load(f"./cache/ubt_seeds/l_{str_map[l_ratio]}_{seed}.pth")
    ul_idx = torch.load(f"./cache/ubt_seeds/ul_{str_map[l_ratio]}_{seed}.pth")

    print("l: ", len(l_idx))
    print("ul: ", len(ul_idx))

    coco_l_set = {}
    coco_u_set = {}

    for idx in tqdm(l_idx):
        try:
            coco_l_set[idx] = coco_full_set[idx]
        except:
            print(idx)

    for idx in tqdm(ul_idx):
        try:
            coco_u_set[idx] = coco_full_set[idx]
        except:
            print(idx)

    print(len(coco_l_set), len(coco_u_set))

    torch.save(coco_l_set, osp.join(cache_path, f"coco17_train{l_ratio}_{seed}.dict"))
    torch.save(coco_u_set, osp.join(cache_path, f"coco17_train{ul_ratio}_{seed}.dict"))
