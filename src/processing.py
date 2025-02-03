import torch
from yolov5.utils.general import non_max_suppression

BOXES_KEY = 'boxes'


def xywh2xyxy(xywh: torch.Tensor) -> torch.Tensor:
    """
    Converts bounding boxes from (x_center, y_center, width, height) to (xmin, ymin, xmax, ymax).

    Args:
        xywh (torch.Tensor): Tensor of shape (N, 4) containing bounding boxes in xywh format.

    Returns:
        torch.Tensor: Converted bounding boxes in xyxy format.
    """
    x_center = xywh[:, 0]
    y_center = xywh[:, 1]
    w = xywh[:, 2]
    h = xywh[:, 3]

    xmin = x_center - w / 2
    ymin = y_center - h / 2
    xmax = x_center + w / 2
    ymax = y_center + h / 2
    return torch.stack([xmin, ymin, xmax, ymax], dim=1)


def process_targets(targets: torch.Tensor, images: torch.Tensor) -> list:
    """
    Converts target annotations to absolute bounding box coordinates and extracts labels.

    Args:
        targets (torch.Tensor): Target annotations (image_index, class_label, x_center, y_center, width, height).
        images (torch.Tensor): Input images of shape (B, C, H, W).

    Returns:
        list: A list of dictionaries with 'boxes' (absolute coordinates) and 'labels' (class labels).
    """
    image_indices = targets[:, 0].unique().int()
    target_list = []
    for idx in image_indices:
        img_targets = targets[targets[:, 0] == idx]
        boxes = xywh2xyxy(img_targets[:, 2:6])
        labels = img_targets[:, 1].int()

        # Get image dimensions
        _, height, width = images[idx].shape  # (C, H, W)

        # Convert boxes from normalized to absolute coordinates
        boxes[:, 0] *= width   # xmin
        boxes[:, 1] *= height  # ymin
        boxes[:, 2] *= width   # xmax
        boxes[:, 3] *= height  # ymax

        target_list.append({BOXES_KEY: boxes, 'labels': labels})
    return target_list


def process_outputs(outputs: torch.Tensor, conf_thres: float, iou_thres: float) -> list:
    """
    Applies non-maximum suppression to model outputs and organizes predictions.

    Args:
        outputs (torch.Tensor): Model outputs.
        conf_thres (float): Confidence threshold for retaining detections.
        iou_thres (float): Intersection-over-Union (IoU) threshold for NMS.

    Returns:
        list[dict]: A list of dictionaries, one for each image. Each dictionary contains:
                    - 'boxes' (torch.Tensor): Detected bounding boxes (xmin, ymin, xmax, ymax).
                    - 'scores' (torch.Tensor): Confidence scores for each detection.
                    - 'labels' (torch.Tensor): Class labels for each detection.
    """
    detections = non_max_suppression(outputs, conf_thres, iou_thres)
    preds = []

    for det in detections:
        if det is not None and len(det):
            boxes = det[:, :4]
            scores = det[:, 4]
            labels = det[:, 5].int()

        else:
            boxes = torch.empty((0, 4), device=outputs[0].device)
            scores = torch.empty((0,), device=outputs[0].device)
            labels_device = outputs[0].device
            labels = torch.empty((0,), dtype=torch.int64, device=labels_device)
        preds.append({BOXES_KEY: boxes, 'scores': scores, 'labels': labels})
    return preds
