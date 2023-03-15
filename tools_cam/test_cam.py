import _init_paths
import json
from timm.models import create_model as create_deit_model
import os
import sys
import pprint
import numpy as np

from config.default import update_config
from config.default import config as cfg
from core.engine import creat_data_loader
from core.functions import prepare_env
from utils import mkdir, Logger
from cams_deit import resize_cam, blend_cam, get_bboxes, cal_iou, draw_bbox
from typing import Dict, List, Optional, Tuple, Union, Sequence
from typing import Dict, List, Optional, Union, Sequence
import numpy as np
import torch
import cv2
from numbers import Number

from tqdm import tqdm
import torch

from models import *

from torch.utils.tensorboard import SummaryWriter
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0
DRAW = False
# VALIDATE_SET = False
topk = (1,5)
cam_curve_interval=.01
multi_contour_eval=False
iou_threshold_list=[30, 50, 70]
RESHAPE_SIZE=224
metrics = ["maxbox_acc", "GT-Known", "loc_acc", "cls_acc"]
OPT_THRED=-1

def creat_model(cfg, args):
    print('==> Preparing networks for baseline...')
    # use gpu
    device = torch.device("cuda")
    assert torch.cuda.is_available(), "CUDA is not available"
    # model and optimizer
    model = create_deit_model(
        cfg.MODEL.ARCH,
        pretrained=True,
        num_classes=cfg.DATA.NUM_CLASSES,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    print(model)
    if args.resume:
        checkpoint = torch.load(args.resume)
        pretrained_dict = {}

        for k, v in checkpoint['state_dict'].items():
            k_ = '.'.join(k.split('.')[1:])

            pretrained_dict.update({k_: v})

        model.load_state_dict(pretrained_dict)
        print('load pretrained ts-cam model.')

    model = torch.nn.DataParallel(
        model, device_ids=list(range(torch.cuda.device_count())))
    model = model.to(device)
    # loss
    cls_criterion = torch.nn.CrossEntropyLoss().to(device)

    print('Preparing networks done!')
    return device, model, cls_criterion


def main():
    args = update_config()

    # root dir
    # create checkpoint directory
    cfg.BASIC.SAVE_DIR = os.path.join(cfg.BASIC.SAVE_ROOT, 'ckpt', cfg.DATA.DATASET, '{}_CAM-NORMAL_SEED{}_BS{}_{}'.format(
        cfg.MODEL.ARCH, cfg.BASIC.SEED, cfg.TRAIN.BATCH_SIZE, cfg.BASIC.TIME))
    cfg.BASIC.ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    log_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'log')
    mkdir(log_dir)
    ckpt_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'ckpt')
    mkdir(ckpt_dir)
    log_file = os.path.join(
        cfg.BASIC.SAVE_DIR, 'Log_' + cfg.BASIC.TIME + '.txt')
    # prepare running environment for the whole project
    prepare_env(cfg)

    # start loging
    sys.stdout = Logger(log_file)
    pprint.pprint(cfg)
    writer = SummaryWriter(log_dir)

    train_loader, test_loader, val_loader = creat_data_loader(
        cfg, os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATA.DATADIR))
    device, model, cls_criterion = creat_model(cfg, args)
    
    if VALIDATE_SET:
        num_imgs = len(val_loader)
        eval_results = val_loc_one_epoch(
            val_loader, model, device)
        opt_thred = eval_results['det_optThred_thr_50.00_top-1']
    else:
        opt_thred = OPT_THRED

    eval_results = val_loc_one_epoch(test_loader, model, device)
    with open(os.path.join(cfg.BASIC.SAVE_DIR, 'test.txt'), 'w') as val_file:
        val_file.write(json.dumps(eval_results))  
    for k, v in eval_results.items():
        if isinstance(v, np.ndarray):
            v = [round(out, 2) for out in v.tolist()]
        elif isinstance(v, Number):
            v = round(v, 2)
        else:
            raise ValueError(f'Unsupport metric type: {type(v)}')
        print(f'\n{k} : {v}')
    if DRAW:
        # opt_thred = eval_results['det_optThred_thr_50.00_top-1']
        opt_thred = 0.35
        print(f"DRAWING IMAGES AT OPTIMAL THRESHOLD {opt_thred}...")
        # draw_bboxes_images(test_loader, model, opt_thred, device, cfg)
        draw_bboxes_images(test_loader, model, opt_thred, device, cfg)


def draw_bboxes_images(dataloader, model, opt_thred, device, cfg):
    print('Start Drawing images ...')
    with torch.no_grad():
        model.eval()
        for _, (input, gt_labels, bbox, image_names, orig) in tqdm(enumerate(dataloader)):
            # update iteration steps
            gt_labels = gt_labels.to(device)
            input = input.to(device)
            orig = orig.cpu().numpy()
            
            _, cams, pred_attn, pred_sem = model(input)
            cams = cams.cpu().tolist()
            pred_attn = pred_attn.cpu().tolist()
            pred_sem = pred_sem.cpu().tolist()
            gt_labels = gt_labels.cpu().tolist()
            inputs = input.cpu().tolist()
            gt_bboxes = [bbox[b].strip().split(' ') for b in range(len(cams))]
            gt_bboxes = [np.array(list(map(float, b))).reshape(-1, 4)
                       for b in gt_bboxes]            
            gt_labels = np.array(gt_labels)
            cams = np.array(cams)  
            pred_sems = np.array(pred_sem)  
            pred_attns = np.array(pred_attn)  
            
                      
            for i in range(len(inputs)):
                img = np.array(orig[i])
                # mean = rearrange(
                #     np.array(list(map(float, cfg.DATA.IMAGE_MEAN))), 'D -> D 1 1')
                # std = rearrange(
                #     np.array(list(map(float, cfg.DATA.IMAGE_STD))), 'D -> D 1 1')
                # img = input*mean+std
                img -= img.min()
                img /= img.max()
                img = img*255
                img = img.transpose(1, 2, 0)
                # to RGB
                img = img[:, :, [2, 1, 0]]

                cam = cams[i]
                pred_sem = pred_sems[i]
                pred_attn = pred_attns[i]
                gt_label = gt_labels[i]
                gt_bbox = gt_bboxes[i]
                
                if len(gt_bbox.shape) == 1:
                    gt_bbox = [gt_bbox]
                
                image_name = image_names[i]
                cam = cam[gt_label, :, :]
                pred_sem = pred_sem[gt_label, :, :]
                pred_attn = pred_attn.squeeze()
                scoremap = resize_cam(cam, size=(
                    cfg.DATA.CROP_SIZE, cfg.DATA.CROP_SIZE))
                boxes_at_thresholds, number_of_box_list = compute_bboxes_from_scoremaps(
                    scoremap=scoremap,
                    scoremap_threshold_list=[opt_thred],
                    multi_contour_eval=multi_contour_eval)
                
                if len(boxes_at_thresholds)==1:
                    boxes_at_thresholds = boxes_at_thresholds[0]
                
                # binary map
                scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)
                _, thr_gray_heatmap = cv2.threshold(
                    src=scoremap_image,
                    thresh=int(opt_thred * np.max(scoremap_image)),
                    maxval=255,
                    type=cv2.THRESH_BINARY)
                
                multiple_iou = calculate_multiple_iou(
                    np.array(boxes_at_thresholds),
                    np.array(gt_bbox))
                
                idx = 0
                sliced_multiple_iou = []
                for nr_box in number_of_box_list:
                    sliced_multiple_iou.append(
                        max(multiple_iou.max(1)[idx:idx + nr_box]))
                    idx += nr_box          
   
                # bbox = get_bboxes(cam, cam_thr=opt_thred)
                blend, heatmap = blend_cam(img, scoremap, bbox)

                iou = sliced_multiple_iou[0]
                boxed_image = draw_bbox(img, iou, np.array(
                    gt_bbox).reshape(-1, 4).astype(np.int32), boxes_at_thresholds, draw_box=True, draw_txt=False)
                roi_image = draw_bbox(blend, iou, np.array(
                    gt_bbox).reshape(-1, 4).astype(np.int32), boxes_at_thresholds, draw_box=False, draw_txt=False)
                
                scoremap = resize_cam(pred_sem, size=(
                    cfg.DATA.CROP_SIZE, cfg.DATA.CROP_SIZE))
                blend, heatmap = blend_cam(img, scoremap, bbox)
                sem_image = draw_bbox(blend, iou, np.array(
                    gt_bbox).reshape(-1, 4).astype(np.int32), boxes_at_thresholds, draw_box=False, draw_txt=False)
                                
                scoremap = resize_cam(pred_attn, size=(
                    cfg.DATA.CROP_SIZE, cfg.DATA.CROP_SIZE))
                blend, heatmap = blend_cam(img, scoremap, bbox)
                attn_image = draw_bbox(blend, iou, np.array(
                    gt_bbox).reshape(-1, 4).astype(np.int32), boxes_at_thresholds, draw_box=False, draw_txt=False)
                                
                                                
                save_dir = os.path.join(
                    cfg.BASIC.SAVE_DIR, 'boxed_image', image_name.split('/')[0])
                save_path = os.path.join(
                    cfg.BASIC.SAVE_DIR, 'boxed_image', image_name)
                mkdir(save_dir)
                
                # print(save_path)
                cv2.imwrite(save_path, boxed_image)
                
                save_dir = os.path.join(
                    cfg.BASIC.SAVE_DIR, 'roi_images', image_name.split('/')[0])
                save_path = os.path.join(
                    cfg.BASIC.SAVE_DIR, 'roi_images', image_name)
                mkdir(save_dir)
                cv2.imwrite(save_path, roi_image)

                save_dir = os.path.join(
                    cfg.BASIC.SAVE_DIR, 'sem_images', image_name.split('/')[0])
                save_path = os.path.join(
                    cfg.BASIC.SAVE_DIR, 'sem_images', image_name)
                mkdir(save_dir)
                cv2.imwrite(save_path, sem_image)
                
                
                save_dir = os.path.join(
                    cfg.BASIC.SAVE_DIR, 'attn_images', image_name.split('/')[0])
                save_path = os.path.join(
                    cfg.BASIC.SAVE_DIR, 'attn_images', image_name)
                mkdir(save_dir)
                cv2.imwrite(save_path, attn_image)
                                                
                save_dir = os.path.join(
                    cfg.BASIC.SAVE_DIR, 'binary', image_name.split('/')[0])
                save_path = os.path.join(
                    cfg.BASIC.SAVE_DIR, 'binary', image_name)
                mkdir(save_dir)
                cv2.imwrite(save_path, thr_gray_heatmap)

def val_loc_one_epoch(val_loader, model, device, opt_thred=-1):
    num_imgs = len(val_loader.dataset)
    cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))
    if opt_thred != -1:
        cam_threshold_list = [opt_thred]
    det_correct = {iou_threshold: np.zeros(
        (len(topk), len(cam_threshold_list))) for iou_threshold in iou_threshold_list}
    if "loc_acc" in metrics:
        loc_correct = {iou_threshold: np.zeros(
            (len(topk), len(cam_threshold_list))) for iou_threshold in iou_threshold_list}
    if "cls_acc" in metrics:
        cls_correct = np.zeros(len(topk))
        
    with torch.no_grad():
        model.eval()
        print('Collect data ... ')
        for _, (input, gt_labels, bbox, _, _) in tqdm(enumerate(val_loader)):
            # update iteration steps
            gt_labels = gt_labels.to(device)
            input = input.to(device)

            cls_scores, cams, _, _ = model(input)
            cls_scores = cls_scores.cpu().tolist()
            cams = cams.cpu().tolist()
            gt_labels = gt_labels.cpu().tolist()

            gt_bboxes = [bbox[b].strip().split(' ') for b in range(len(cams))]
            gt_bboxes = [np.array(list(map(float, b))).reshape(-1, 4)
                       for b in gt_bboxes]

            cls_scores = np.array(cls_scores)
            gt_labels = np.array(gt_labels)
            cams = np.array(cams)
            
            max_k = np.max(topk).item()  # naive int
            topk_ind = torch.topk(torch.from_numpy(
                cls_scores), max_k)[-1].numpy()  # [B K]
            cams = np.array([np.take(a, idx, axis=0) for (a, idx) in zip(
                cams, topk_ind)])  # index for each batch # [B topk H W]
            pred_labels = topk_ind  # [B K]

            for i in range(cams.shape[0]):
                count(cams[i], cam_threshold_list, gt_bboxes[i], pred_labels[i], gt_labels[i], det_correct, loc_correct, cls_correct)
            
    return evaluate((det_correct, loc_correct, cls_correct, num_imgs))


def evaluate(
    results: Tuple,
    metric: Union[str, List[str]] = [
        "maxbox_acc", "GT-Known", "loc_acc", "cls_acc"],
    metric_options: Optional[dict] = None,
    opt_thred=-1,
) -> Dict:
    
    """
    Evaluate the dataset.
    Args:
        results cls_scores, bboxes, gt_bbox): Testing results of the dataset.
        metric (str | list[str]): Metrics to be evaluated.
            Default value is `accuracy`.
        metric_options (dict | None): Options for calculating metrics.
            Allowed keys are 'topk', 'thrs' and 'average_mode'.
            Default: None.
        logger (logging.Logger | None): Logger used for printing
            related information during evaluation. Default: None.
    Returns:
        dict: evaluation results
    """

    if isinstance(metric, str):
        metrics = [metric]
    else:
        metrics = metric
    allowed_metrics = ["cls_acc", "maxbox_acc", "GT-Known", "loc_acc"]

    eval_results = {}

    det_correct, loc_correct, cls_correct, num_imgs = results
    # remove padding values -1
    # list[B n array(N*4)] where N is the padded length of arraries -> n is the number of cases per batch
    # -> n` is the number of gt_bboxes per case
    # list[(B*n) array(n`*4)]
    # -> [B 4] # for CUB-200-2011 only one box is available for each case

    invalid_metrics = set(metrics) - set(allowed_metrics)
    if len(invalid_metrics) != 0:
        raise ValueError(f"metirc {invalid_metrics} is not supported.")

    if metric_options is None:
        metric_options = {"topk": (1, 5)}
    topk = metric_options.get("topk", (1, 5))
    thrs = metric_options.get("thrs", 0.0)

    if "cls_acc" in metrics:
        # [top]
        cls_acc = cls_correct * 100. / float(num_imgs)
        if isinstance(topk, tuple):
            eval_results_ = {
                f"cls_acc_top-{k}": a for k, a in zip(topk, cls_acc)}
        else:
            eval_results_ = {"cls_acc": cls_acc}
        if isinstance(thrs, tuple):
            for key, values in eval_results_.items():
                eval_results.update(
                    {
                        f"{key}_thr_{thr:.2f}": value.item()
                        for thr, value in zip(thrs, values)
                    }
                )
        else:
            eval_results.update({k: v.item()
                                for k, v in eval_results_.items()})

    if any([i in metrics for i in ["maxbox_acc", "GT-Known", "loc_acc"]]):

        if any(i in metrics for i in ["GT-Known", "maxbox_acc"]):
            max_box_acc = []  # [len(iou_list), top]
            opt_threds = []  # [len(iou_list), top]
            for _THRESHOLD in iou_threshold_list:
                # [top, len(threds)]
                localization_accuracies = det_correct[_THRESHOLD] * 100. / float(
                    num_imgs)
                opt_thred = np.argmax(
                    localization_accuracies, axis=1) * cam_curve_interval  # [top]
                max_box_acc.append(localization_accuracies.max(1))
                opt_threds.append(opt_thred)

            if "GT-Known" in metrics:
                GTK = max_box_acc[iou_threshold_list.index(50)]  # [top]
                if isinstance(topk, tuple):
                    eval_results_ = {f"GT-Known_top-{k}": a for k,
                                     a in zip(topk, GTK)}
                else:
                    eval_results_ = {"GT-Known": GTK}
                eval_results.update({k: v.item()
                                    for k, v in eval_results_.items()})
            if "maxbox_acc" in metrics:
                if isinstance(iou_threshold_list, Sequence):
                    for thr in iou_threshold_list:
                        # [top]
                        mba = max_box_acc[iou_threshold_list.index(thr)]
                        if isinstance(topk, tuple):
                            eval_results_ = {f"maxbox_acc_thr_{thr:.2f}_top-{k}": a for k,
                                             a in zip(topk, mba)}
                            eval_results.update(
                                {k: v.item() for k, v in eval_results_.items()})
                        else:
                            eval_results.update(
                                {f"maxbox_acc_thr_{thr:.2f}": mba})
                else:
                    thr = iou_threshold_list
                    mba = max_box_acc[iou_threshold_list.index(thr)]
                    if isinstance(topk, tuple):
                        if isinstance(topk, tuple):
                            eval_results_ = {f"maxbox_acc_thr_{thr:.2f}_top-{k}": a for k,
                                             a in zip(topk, mba)}
                            eval_results.update(
                                {k: v.item() for k, v in eval_results_.items()})
                        else:
                            eval_results.update(
                                {f"maxbox_acc_thr_{thr:.2f}": mba})

            # update threds
            if isinstance(iou_threshold_list, Sequence):
                for thr in iou_threshold_list:
                    # [top]
                    opts = opt_threds[iou_threshold_list.index(thr)]
                    if isinstance(topk, tuple):
                        eval_results_ = {f"det_optThred_thr_{thr:.2f}_top-{k}": a for k,
                                         a in zip(topk, opts)}
                        eval_results.update(
                            {k: v.item() for k, v in eval_results_.items()})
                    else:
                        eval_results.update(
                            {f"det_optThred_thr_{thr:.2f}": opts})
            else:
                thr = iou_threshold_list
                opts = opt_threds[iou_threshold_list.index(thr)]
                if isinstance(topk, tuple):
                    if isinstance(topk, tuple):
                        eval_results_ = {f"det_optThred_thr_{thr:.2f}_top-{k}": a for k,
                                         a in zip(topk, opts)}
                        eval_results.update(
                            {k: v.item() for k, v in eval_results_.items()})
                    else:
                        eval_results.update(
                            {f"det_optThred_thr_{thr:.2f}": opts})

    if "loc_acc" in metrics:
        loc_acc = []  # [len(iou_list), top]
        opt_threds = []  # [len(iou_list), top]
        for _THRESHOLD in iou_threshold_list:
            # [top, len(threds)]
            localization_accuracies = loc_correct[_THRESHOLD] * \
                100. / float(num_imgs)
            opt_thred = np.argmax(
                localization_accuracies, axis=1) * cam_curve_interval  # [top]
            loc_acc.append(localization_accuracies.max(1))
            opt_threds.append(opt_thred)

        if isinstance(iou_threshold_list, Sequence):
            for thr in iou_threshold_list:
                acc = loc_acc[iou_threshold_list.index(thr)]  # [top]
                if isinstance(topk, tuple):
                    eval_results_ = {f"loc_acc_thr_{thr:.2f}_top-{k}": a for k,
                                     a in zip(topk, acc)}
                    eval_results.update({k: v.item()
                                        for k, v in eval_results_.items()})
                else:
                    eval_results.update({f"loc_acc_thr_{thr:.2f}": acc})
        else:
            thr = iou_threshold_list
            acc = loc_acc[iou_threshold_list.index(thr)]
            if isinstance(topk, tuple):
                if isinstance(topk, tuple):
                    eval_results_ = {f"loc_acc_thr_{thr:.2f}_top-{k}": a for k,
                                     a in zip(topk, acc)}
                    eval_results.update({k: v.item()
                                        for k, v in eval_results_.items()})
                else:
                    eval_results.update({f"loc_acc_thr_{thr:.2f}": acc})
        # update threds
        if isinstance(iou_threshold_list, Sequence):
            for thr in iou_threshold_list:
                # [top]
                opts = opt_threds[iou_threshold_list.index(thr)]
                if isinstance(topk, tuple):
                    eval_results_ = {f"loc_optThred_thr_{thr:.2f}_top-{k}": a for k,
                                     a in zip(topk, opts)}
                    eval_results.update(
                        {k: v.item() for k, v in eval_results_.items()})
                else:
                    eval_results.update(
                        {f"loc_optThred_thr_{thr:.2f}": opts})
        else:
            thr = iou_threshold_list
            opts = opt_threds[iou_threshold_list.index(thr)]
            if isinstance(topk, tuple):
                if isinstance(topk, tuple):
                    eval_results_ = {f"det_optThred_thr_{thr:.2f}_top-{k}": a for k,
                                     a in zip(topk, opts)}
                    eval_results.update(
                        {k: v.item() for k, v in eval_results_.items()})
                else:
                    eval_results.update(
                        {f"det_optThred_thr_{thr:.2f}": opts})

    return eval_results


def resizeNorm(cam, size=(224, 224)):
    cam = cv2.resize(cam, (size[0], size[1]))
    cam_min, cam_max = cam.min(), cam.max()
    cam = (cam - cam_min) / (cam_max - cam_min)
    return cam


def count(cams,
          cam_threshold_list,
          gt_bboxes,
          pred_labels,
          gt_labels,
          det_correct,
          loc_correct,
          cls_correct,
          ):
    # cams -> [topk H W] j -> topk
    # init count array for saving results
    max_k = np.max(topk).item()
    cnt_det = {iou_threshold: np.zeros(
        (max_k, len(cam_threshold_list))) for iou_threshold in iou_threshold_list}
    cnt_loc = {iou_threshold: np.zeros(
        (max_k, len(cam_threshold_list))) for iou_threshold in iou_threshold_list}
    cnt_cls = np.zeros(max_k)
    
    for j in range(cams.shape[0]): # j -> topk
        scoremap = cams[j]
        scoremap = resizeNorm(scoremap, (RESHAPE_SIZE, RESHAPE_SIZE))
        # apdated from https://github.com/clovaai/wsolevaluation#prepare-heatmaps-to-evaluate
        boxes_at_thresholds, number_of_box_list = compute_bboxes_from_scoremaps(
            scoremap=scoremap,
            scoremap_threshold_list=cam_threshold_list,
            multi_contour_eval=multi_contour_eval)

        boxes_at_thresholds = np.concatenate(
            boxes_at_thresholds, axis=0)

        multiple_iou = calculate_multiple_iou(
            np.array(boxes_at_thresholds),
            np.array(gt_bboxes))

        idx = 0
        sliced_multiple_iou = []
        for nr_box in number_of_box_list:
            sliced_multiple_iou.append(
                max(multiple_iou.max(1)[idx:idx + nr_box]))
            idx += nr_box

        for _THRESHOLD in iou_threshold_list:
            det_arry = cnt_det[_THRESHOLD]
            correct_threshold_indices = \
                np.where(np.asarray(sliced_multiple_iou)
                         >= (_THRESHOLD/100))[0]
            det_arry[j][correct_threshold_indices] += 1
            if "loc_acc" in metrics and pred_labels[j] == gt_labels:
                loc_arry = cnt_loc[_THRESHOLD]
                loc_arry[j][correct_threshold_indices] += 1

        if 'cls_acc' in metrics and pred_labels[j] == gt_labels:
            cnt_cls[j] += 1
            
    # calculate top k
    for i, _THRESHOLD in enumerate(iou_threshold_list):
        thr_det = det_correct[_THRESHOLD]  # [tops threds]
        for idx, k in enumerate(topk):
            counts = np.sum(cnt_det[_THRESHOLD][:k], axis=0) > 0
            counts = counts.astype(int)
            thr_det[idx] += counts
        if "loc_acc" in metrics:
            thr_loc = loc_correct[_THRESHOLD]
            for idx, k in enumerate(topk):
                counts = np.sum(cnt_loc[_THRESHOLD][:k], axis=0) > 0
                counts = counts.astype(int)
                thr_loc[idx] += counts

    if 'cls_acc' in metrics:
        for idx, k in enumerate(topk):
            counts = np.sum(cnt_cls[:k], axis=0) > 0
            counts = counts.astype(int)
            cls_correct[idx] += counts

def compute_bboxes_from_scoremaps(scoremap, scoremap_threshold_list,
                                  multi_contour_eval=False):
    """
    Args:
        scoremap: numpy.ndarray(dtype=np.float32, size=(H, W)) between 0 and 1
        scoremap_threshold_list: iterable
        multi_contour_eval: flag for multi-contour evaluation
    Returns:
        estimated_boxes_at_each_thr: list of estimated boxes (list of np.array)
            at each cam threshold
        number_of_box_list: list of the number of boxes at each cam threshold
    """
    check_scoremap_validity(scoremap)
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)

    def scoremap2bbox(threshold):
        _, thr_gray_heatmap = cv2.threshold(
            src=scoremap_image,
            thresh=int(threshold * np.max(scoremap_image)),
            maxval=255,
            type=cv2.THRESH_BINARY)
        contours = cv2.findContours(
            image=thr_gray_heatmap,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

        if len(contours) == 0:
            return np.asarray([[0, 0, 0, 0]]), 1

        if not multi_contour_eval:
            contours = [max(contours, key=cv2.contourArea)]

        estimated_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)
            estimated_boxes.append([x0, y0, x1, y1])

        return np.asarray(estimated_boxes), len(contours)

    estimated_boxes_at_each_thr = []
    number_of_box_list = []
    for threshold in scoremap_threshold_list:
        boxes, number_of_box = scoremap2bbox(threshold)
        estimated_boxes_at_each_thr.append(boxes)
        number_of_box_list.append(number_of_box)

    return estimated_boxes_at_each_thr, number_of_box_list


def check_scoremap_validity(scoremap):
    if not isinstance(scoremap, np.ndarray):
        raise TypeError("Scoremap must be a numpy array; it is {}."
                        .format(type(scoremap)))
    if len(scoremap.shape) != 2:
        raise ValueError("Scoremap must be a 2D array; it is {}D."
                         .format(len(scoremap.shape)))
    if np.isnan(scoremap).any():
        raise ValueError("Scoremap must not contain nans.")
    if (scoremap > 1).any() or (scoremap < 0).any():
        raise ValueError("Scoremap must be in range [0, 1]."
                         "scoremap.min()={}, scoremap.max()={}."
                         .format(scoremap.min(), scoremap.max()))


def calculate_multiple_iou(box_a, box_b):
    """
    Args:
        box_a: numpy.ndarray(dtype=np.int, shape=(num_a, 4))
            x0y0x1y1 convention.
        box_b: numpy.ndarray(dtype=np.int, shape=(num_b, 4))
            x0y0x1y1 convention.
    Returns:
        ious: numpy.ndarray(dtype=np.int, shape(num_a, num_b))
    """
    num_a = box_a.shape[0]
    num_b = box_b.shape[0]

    check_box_convention(box_a, 'x0y0x1y1')
    check_box_convention(box_b, 'x0y0x1y1')

    # num_a x 4 -> num_a x num_b x 4
    box_a = np.tile(box_a, num_b)
    box_a = np.expand_dims(box_a, axis=1).reshape((num_a, num_b, -1))

    # num_b x 4 -> num_b x num_a x 4
    box_b = np.tile(box_b, num_a)
    box_b = np.expand_dims(box_b, axis=1).reshape((num_b, num_a, -1))

    # num_b x num_a x 4 -> num_a x num_b x 4
    box_b = np.transpose(box_b, (1, 0, 2))

    # num_a x num_b
    min_x = np.maximum(box_a[:, :, 0], box_b[:, :, 0])
    min_y = np.maximum(box_a[:, :, 1], box_b[:, :, 1])
    max_x = np.minimum(box_a[:, :, 2], box_b[:, :, 2])
    max_y = np.minimum(box_a[:, :, 3], box_b[:, :, 3])

    # num_a x num_b
    area_intersect = (np.maximum(0, max_x - min_x + 1)
                      * np.maximum(0, max_y - min_y + 1))
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0] + 1) *
              (box_a[:, :, 3] - box_a[:, :, 1] + 1))
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0] + 1) *
              (box_b[:, :, 3] - box_b[:, :, 1] + 1))

    denominator = area_a + area_b - area_intersect
    degenerate_indices = np.where(denominator <= 0)
    denominator[degenerate_indices] = 1

    ious = area_intersect / denominator
    ious[degenerate_indices] = 0
    return ious


def check_box_convention(boxes, convention):
    """
    Args:
        boxes: numpy.ndarray(dtype=np.int or np.float, shape=(num_boxes, 4))
        convention: string. One of ['x0y0x1y1', 'xywh'].
    Raises:
        RuntimeError if box does not meet the convention.
    """
    if (boxes < 0).any():
        raise RuntimeError("Box coordinates must be non-negative.")

    if len(boxes.shape) == 1:
        boxes = np.expand_dims(boxes, 0)
    elif len(boxes.shape) != 2:
        raise RuntimeError("Box array must have dimension (4) or "
                           "(num_boxes, 4).")

    if boxes.shape[1] != 4:
        raise RuntimeError("Box array must have dimension (4) or "
                           "(num_boxes, 4).")

    if convention == 'x0y0x1y1':
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
    elif convention == 'xywh':
        widths = boxes[:, 2]
        heights = boxes[:, 3]
    else:
        raise ValueError("Unknown convention {}.".format(convention))

    if (widths < 0).any() or (heights < 0).any():
        raise RuntimeError("Boxes do not follow the {} convention."
                           .format(convention))


if __name__ == "__main__":
    main()
