import numpy as np
import torch


def area_of(left_top, right_bottom) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def compute_average_precision(precision, recall):
    """
    It computes average precision based on the definition of Pascal Competition. It computes the under curve area
    of precision and recall. Recall follows the normal definition. Precision is a variant.
    pascal_precision[i] = typical_precision[i:].max()
    """
    # identical but faster version of new_precision[i] = old_precision[i:].max()
    precision = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    # find the index where the value changes
    recall = np.concatenate([[0.0], recall, [1.0]])
    changing_points = np.where(recall[1:] != recall[:-1])[0]

    # compute under curve area
    areas = (recall[changing_points + 1] - recall[changing_points]) * precision[changing_points + 1]
    return areas.sum()


def compute_voc2007_average_precision(precision, recall):
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11.
    return ap


def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric,
                                        name="", plot=True):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    if use_2007_metric:
        ap = compute_voc2007_average_precision(precision, recall)
    else:
        ap = compute_average_precision(precision, recall)
    if plot:
        sorted_scores = scores[sorted_indexes]
        plot_map_recall_precision(ap, recall, precision, sorted_scores, name)
    return ap


def plot_map_recall_precision(ap, recall, precision, sorted_scores, name):
    """
    sorted_scores = scores[sorted_indexes]
    :param ap:
    :param recall:
    :param precision:
    :param sorted_scores:
    :param name:
    :return:
    """
    import matplotlib.pyplot as plt
    # 逆序排序
    recall = [0] + recall.tolist() + [1]
    precision = [1] + precision.tolist() + [0]
    sorted_scores = [1] + sorted_scores.tolist() + [0]
    plt.figure()
    plt.plot(recall, precision, label='Precision')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    ap_str = "{0:.2f}%".format(ap * 100)
    # ap_str = "{0:.4f}%".format(ap * 100)
    plt.title('Precision x Recall curve \nClass: %s, AP: %s' % (name, ap_str))
    # plt.title('Precision x Recall curve \nClass: %s' % str(classId))
    plt.legend(shadow=True)
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.grid()

    plt.figure()
    plt.plot(sorted_scores, recall, label='Recall')
    plt.xlabel('Scores')
    plt.ylabel('Recall')
    plt.title('Recall-Scores\nClass:{}'.format(name))
    # plt.title('Precision x Recall curve \nClass: %s' % str(classId))
    plt.legend(shadow=True)
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.grid()

    plt.figure()
    plt.plot(sorted_scores, precision, label='Precision')
    plt.xlabel('Scores')
    plt.ylabel('Precision')
    plt.title('Precision-Scores\nClass:{}'.format(name))
    # plt.title('Precision x Recall curve \nClass: %s' % str(classId))
    plt.legend(shadow=True)
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.grid()

    plt.show()
    # plt.waitforbuttonpress()
    plt.pause(0.05)
