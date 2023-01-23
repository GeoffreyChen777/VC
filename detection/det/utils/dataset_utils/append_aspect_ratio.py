import torch
from PIL import Image
import os
from tqdm import tqdm

for path in os.listdir("./cache"):
    if path.endswith(".dict"):
        print(path)
        cache = torch.load("./cache/"+path)

        new_cache = {}
        for id, value in tqdm(cache.items()):
            img = Image.open(value["image_path"])
            if img.size[1] > img.size[0]:
                group = 1
            else:
                group = 0

            value["aspect_ratio"] = group
            new_cache[id] = value

        torch.save(new_cache, "./cache/new/" + path)