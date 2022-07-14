from PIL import ImageFont, ImageDraw
from dataset.transforms import Denormalize
from torchvision.transforms.functional import to_pil_image
from utils.meta import *
from gvcore.utils.structure import GenericData
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    def __init__(self, meta=COCOMeta, form="x1,y1,x2,y2,cls,score", t=0.5, t_key="score", show_tag=True):
        self.meta = meta()
        self.form = form
        self.t = t
        self.t_key = t_key
        self.show_tag = show_tag

    @staticmethod
    def draw_txt(draw, x, y, text, font):
        draw.text((x + 1, y + 1), text, (50, 50, 50), font=font)
        draw.text((x, y), text, (255, 255, 255), font=font)

    def __call__(self, img, bboxes, form=None, denorm=True):
        if denorm:
            img = Denormalize()(img) / 255.0
        img = to_pil_image(img.detach().cpu())
        draw = ImageDraw.Draw(img)
        bboxes = bboxes.detach().cpu()

        form = self.form if form is None else form
        split_form = form.split(",")
        form_idx = {}
        score_keys = []
        for i, key in enumerate(split_form):
            form_idx[key] = i
            if key not in "_,x,y,w,h,x1,y1,x2,y2,cls":
                score_keys.append(key)

        for box in bboxes:
            if "x,y,w,h" in form:
                x1 = box[form_idx["x"]] - int(box[form_idx["w"]] / 2)
                y1 = box[form_idx["y"]] - int(box[form_idx["h"]] / 2)
                x2 = box[form_idx["x"]] + int(box[form_idx["w"]] / 2)
                y2 = box[form_idx["y"]] + int(box[form_idx["h"]] / 2)
            elif "x1,y1,x2,y2" in form:
                x1 = box[form_idx["x1"]]
                y1 = box[form_idx["y1"]]
                x2 = box[form_idx["x2"]]
                y2 = box[form_idx["y2"]]
            else:
                raise ValueError

            if "score" in form_idx:
                score = box[form_idx["score"]]
                if score < self.t:
                    continue

            if "cls" in form_idx:
                cls = int(box[form_idx["cls"]])
            else:
                cls = -1

            meta = self.meta(cls)
            color = tuple(meta["color"])
            cls_name = meta["name"]

            draw.rectangle(xy=[x1, y1, x2, y2], outline=color)

            if self.show_tag:
                draw.rectangle(xy=[x1, y1, x1 + 14 + 8 * len(cls_name) + 38 * len(score_keys), y1 - 18], fill=color)
                monospace = ImageFont.truetype("./assets/ShareTechMono.ttf", 14)

                self.draw_txt(draw, x1 + 5, y1 - 17, cls_name, monospace)

                for i, score_key in enumerate(score_keys):
                    self.draw_txt(
                        draw,
                        x1 + 10 + 8 * len(cls_name) + i * 38,
                        y1 - 17,
                        "{:.2f}".format(box[form_idx[score_key]]),
                        monospace,
                    )

        return img


def vis(data_list, result_list=None, out_path=None, form="x1,y1,x2,y2,cls", threshold=0, meta=COCOMeta):

    visualizer = Visualizer(t=threshold, meta=meta)

    if isinstance(data_list, GenericData):
        data_list = [data_list]
        result_list = [result_list]
    elif isinstance(data_list, list):
        pass
    else:
        raise ValueError("data_list need to be a GenericData object or list of GenericData.")

    if result_list is None:
        result_list = [result_list] * len(data_list)

    imgs = []
    for i, (data, result) in enumerate(zip(data_list, result_list)):
        img = data.img.cpu()
        if result is not None:
            box = result.cpu()
        else:
            box = data.label.cpu()
        img = visualizer(img, box, form=form)
        img = np.asarray(img)
        imgs.append(img)

    imgs = np.concatenate(imgs, axis=0)

    plt.figure(figsize=(imgs.shape[1] / 200, imgs.shape[0] / 200), dpi=200)
    plt.subplots_adjust(bottom=0, top=1, left=0, right=1)
    plt.imshow(imgs)

    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)
