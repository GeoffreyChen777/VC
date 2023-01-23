import random
import torchvision.io as io
import torch


ratio = 80
fold = 2

index = []
with open("./data/cityscapes/index/train.txt", "r") as f:
    lines = f.readlines()
    for i in lines:
        index.append(i.strip())

n = int(1 / ratio * len(index))

random.shuffle(index)


with open(f"./data/cityscapes/index/l.1.{ratio}.fold{fold}.txt", "w") as f:
    for i in index[:n]:
        f.write(i + "\n")

with open(f"./data/cityscapes/index/u.1.{ratio}.fold{fold}.txt", "w") as f:
    for i in index[n:]:
        f.write(i + "\n")


with open(f"./data/cityscapes/index/l.1.{ratio}.fold{fold}.txt", "r") as f:
    index = f.readlines()
    index = [i.strip() for i in index]


counts = torch.zeros(256)

for i in index:
    label = io.read_image(f"./data/cityscapes/gtFine/{i}_gtFine_labelTrainIds.png")
    cls, count = label.unique(return_counts=True)
    counts = torch.index_add(counts, 0, cls.long(), count.float())

print(counts[:19] / counts[:19].sum())
