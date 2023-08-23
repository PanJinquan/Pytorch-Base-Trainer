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
    if isinstance(boxes0, np.ndarray):
        boxes0 = torch.from_numpy(boxes0)
    if isinstance(boxes1, np.ndarray):
        boxes1 = torch.from_numpy(boxes1)
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


def compute_average_precision_per_class(gt_boxes,
                                        pred_result: dict,
                                        iou_threshold,
                                        use_2007_metric,
                                        name="",
                                        plot=False):
    """
    :param num_true_cases:
    :param pred_result:
    :param iou_threshold:
    :param use_2007_metric:
    :param name:
    :param plot:
    :return:
    """
    num_true_cases = [len(gt) for gt in gt_boxes.values()]
    num_true_cases = sum(num_true_cases)
    image_ids = []
    boxes = []
    scores = []
    for file, item in pred_result.items():
        for line in item:
            image_ids.append(file)
            scores.append(line[0])
            boxes.append(line[1:])
    if len(scores) == 0:
        result = {"ap": 0}
        return result
    scores = np.asarray(scores, dtype=np.float32)
    sorted_indexes = np.argsort(-scores)
    boxes = [boxes[i] for i in sorted_indexes]
    image_ids = [image_ids[i] for i in sorted_indexes]
    true_positive = np.zeros(len(image_ids), dtype=np.float32)
    false_positive = np.zeros(len(image_ids), dtype=np.float32)
    matched = set()
    for i, image_id in enumerate(image_ids):
        box = np.asarray(boxes[i])
        if image_id not in gt_boxes:
            false_positive[i] = 1
            continue

        gt_box = np.asarray(gt_boxes[image_id])
        ious = iou_of(box, gt_box)
        max_iou = torch.max(ious).item()
        max_arg = torch.argmax(ious).item()
        if max_iou > iou_threshold:
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
    sorted_scores = scores[sorted_indexes]
    result = {"ap": ap,
              "recall": recall,
              "precision": precision,
              "sorted_scores": sorted_scores,
              "true_positive": true_positive,
              "false_positive": false_positive,
              "num_true_cases": num_true_cases
              }

    if plot:
        plot_ap_recall_precision(ap, recall, precision, sorted_scores, name)
    # print_info(result, name)
    return result


def print_info(result, name, target_scores=0.5):
    """
    :param result:
    :param name:
    :param target_scores:
    :return:
    """
    info = "--" * 20
    ap = result["ap"]
    sorted_scores = result["sorted_scores"]
    true_positive = result["true_positive"]
    false_positive = result["false_positive"]
    num_true_cases = result["num_true_cases"]
    recall = result["recall"]
    precision = result["precision"]
    index = get_nearest_neighbor(sorted_scores, target=target_scores, k=1)
    tp_fp = true_positive + false_positive
    info += "\nscores:{}".format(sorted_scores[index])
    info += "\nground true:{}".format(num_true_cases)
    info += "\ntrue_positive:{}".format(true_positive[index])
    info += "\nfalse_positive:{}".format(false_positive[index])
    info += "\ntp_fp:{}".format(tp_fp[index])

    info += "\nrecall:{}".format(recall[index])
    info += "\nprecision:{}".format(precision[index])
    info += "\n{} AP:{}\n".format(name, ap)
    info += "--" * 20
    print(info)
    return info


def get_nearest_neighbor(data, target, k=1):
    """
    :param data:
    :param target:
    :param k:
    :return:
    """
    assert k == 1
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    diff = np.abs(data - target)
    index = np.argmin(diff)
    return index


def plot_map_recall_precision(map_result: dict):
    """
    :param map_result:
    :return:
    """
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')
    class_name = list(map_result.keys())

    # Precision-Recall
    plt.figure()
    aps = []
    for name in class_name:
        recall = map_result[name]["recall"]
        precision = map_result[name]["precision"]
        ap = map_result[name]["ap"]
        recall = [0] + recall.tolist() + [1]
        precision = [1] + precision.tolist() + [0]
        plt.plot(recall, precision, label='{} AP:{:3.5f}%'.format(name, ap * 100))
        aps.append(ap)
    map = sum(aps) / len(aps)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve\nmAP={:3.5f}'.format(map))
    plt.legend(shadow=True)
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.grid()

    # Recall-Scores
    plt.figure()
    for name in class_name:
        recall = map_result[name]["recall"]
        sorted_scores = map_result[name]["sorted_scores"]
        recall = [0] + recall.tolist() + [1]
        sorted_scores = [1] + sorted_scores.tolist() + [0]
        plt.plot(sorted_scores, recall, label='{} Recall'.format(name))
    plt.xlabel('Scores')
    plt.ylabel('Recall')
    plt.title('Recall-Scores')
    # plt.title('Precision x Recall curve \nClass: %s' % str(classId))
    plt.legend(shadow=True)
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.grid()
    # Precision-Scores
    plt.figure()
    for name in class_name:
        precision = map_result[name]["precision"]
        sorted_scores = map_result[name]["sorted_scores"]
        precision = [1] + precision.tolist() + [0]
        sorted_scores = [1] + sorted_scores.tolist() + [0]
        plt.plot(sorted_scores, precision, label='{} Precision'.format(name))
    plt.xlabel('Scores')
    plt.ylabel('Precision')
    plt.title('Precision-Scores')
    plt.legend(shadow=True)
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.grid()
    plt.show()


def plot_ap_recall_precision(ap, recall, precision, sorted_scores, name):
    """
    sorted_scores = scores[sorted_indexes]
    :param ap:
    :param recall:
    :param precision:
    :param sorted_scores:
    :param name:
    :return:
    """
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')

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
    plt.title('Precision x Recall curve \nClass: {}, AP: {:3.5f}'.format(name, ap_str))
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
