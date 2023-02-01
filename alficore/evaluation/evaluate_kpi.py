import numpy as np
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


def calculate_sdc1(predicted,groundtruth):
    predicted_top = predicted[:,0]
    return (len(predicted_top) - (len(set(predicted_top) & set(groundtruth))))/len(predicted_top)*100

def calculate_sdc5(predicted,groundtruth,numclasses):
    if numclasses < 5:
        return None
    count = 0
    for x,y in zip(predicted,groundtruth):
        if y not in x:
            count = count + 1

    return count/len(groundtruth) * 100

def calculate_sdc10(confidences,groundtruth_confidences):
    count = 0
    for x,y in zip(confidences,groundtruth_confidences):
        if abs(x-y) > 0.1:
            count = count + 1
    return count/len(confidences) * 100

def calculate_sdc20(confidences,groundtruth_confidences):
    count = 0
    for x,y in zip(confidences,groundtruth_confidences):
        if abs(x-y) > 0.2:
            count = count + 1
    return count/len(confidences) * 100

def calculate_sdcs(predicted,groundtruth,numclasses,confidences,groundtruth_confidences):
    x = calculate_sdc1(predicted,groundtruth)
    y = calculate_sdc5(predicted,groundtruth,numclasses)
    z = calculate_sdc10(confidences,groundtruth_confidences)
    w = calculate_sdc20(confidences,groundtruth_confidences)
    return calculate_sdc1(predicted,groundtruth) , calculate_sdc5(predicted,groundtruth,numclasses), calculate_sdc10(confidences,groundtruth_confidences), calculate_sdc20(confidences,groundtruth_confidences)