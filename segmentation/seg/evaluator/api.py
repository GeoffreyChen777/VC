import torch
import torchvision.transforms.functional as tvF


def intersect_and_union(pred, label, num_classes, ignore_index, label_map=dict(), reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """
    if label.dim() == 3:
        label = label.squeeze(0)

    if pred.shape != label.shape:
        pred = tvF.resize(pred.unsqueeze(0), label.shape[-2:], interpolation=tvF.InterpolationMode.NEAREST).squeeze(0)

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = label != ignore_index
    pred = pred[mask]
    label = label[mask]

    intersect = pred[pred == label]
    area_intersect = torch.histc(intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred = torch.histc(pred.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred + area_label - area_intersect
    return area_intersect, area_union, area_pred, area_label
