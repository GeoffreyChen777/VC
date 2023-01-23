import random
import torchvision.io as io
import torch


ratio = 128
fold = 3

index = []
with open("./data/VOCSegAug/index/train.txt", "r") as f:
    lines = f.readlines()
    for i in lines:
        index.append(i.strip())

n = int(1 / ratio * len(index))

random.shuffle(index)


with open(f"./data/VOCSegAug/index/l.1.{ratio}.fold{fold}.txt", "w") as f:
    for i in index[:n]:
        f.write(i + "\n")

with open(f"./data/VOCSegAug/index/u.1.{ratio}.fold{fold}.txt", "w") as f:
    for i in index[n:]:
        f.write(i + "\n")


with open(f"./data/VOCSegAug/index/l.1.{ratio}.fold{fold}.txt", "r") as f:
    index = f.readlines()
    index = [i.strip() for i in index]


counts = torch.zeros(256)

for i in index:
    label = io.read_image(f"./data/VOCSegAug/label/{i}.png")
    cls, count = label.unique(return_counts=True)
    counts = torch.index_add(counts, 0, cls.long(), count.float())

print(counts[:21] / counts[:21].sum())
