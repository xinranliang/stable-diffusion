import sys, os
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import numpy as np
import json, cv2, random

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from utils import SimpleDataset, social_job_list

# guidance values
w_lst = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_area(image, predictor):
    outputs = predictor(cv2.imread(image))
    pred_labels = outputs["instances"].pred_classes

    if len(pred_labels) > 0 and torch.min(pred_labels) == 0:
        correct_boxes, correct_masks = [], []

        num_total_labels = len(outputs["instances"].pred_classes)

        for index in range(num_total_labels):
            if pred_labels[index] == 0:
                correct_boxes.append(outputs["instances"].pred_boxes[index])
                correct_masks.append(outputs["instances"].pred_masks[index])

        box_areas = [each_box.area().detach().cpu().item() for each_box in correct_boxes]
        box_area = max(box_areas)
        mask_areas = [torch.sum(each_mask).detach().cpu().item() for each_mask in correct_masks]
        mask_area = max(mask_areas)
    else:
        assert (len(pred_labels) == 0 or torch.min(pred_labels) >= 1)
        box_area = 0.0
        mask_area = 0.0
    
    return box_area, mask_area

def visualize_area(image, outputs):
    # visualize utils
    im = cv2.imread(image)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(image.replace(".png", "_viz.png"), out.get_image()[:, :, ::-1])


def compute_area_metrics(args):
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    # overall
    for cfg_w in w_lst:
        box_area, mask_area = [], []
        print(f"sampling from stable-diffusion-v2 w/ cfg_w = {cfg_w}")
        img_dataset = SimpleDataset(root=f"{args.master_folder}/guide_w{cfg_w}", subset=args.subset_name, exp_date=args.date)
        img_dataload = DataLoader(img_dataset, batch_size=1, shuffle=False, num_workers=4) # per-image inference
        for sample, path in iter(img_dataload):
            with torch.no_grad():
                new_box_area, new_mask_area = compute_area(path[0], predictor)
            box_area.append(new_box_area)
            mask_area.append(new_mask_area)
        print(f"total number of samples: {len(box_area)}")
        box_area_avg = np.mean(np.array(box_area, dtype=np.float64))
        mask_area_avg = np.mean(np.array(mask_area, dtype=np.float64))
        print(f"mean detected box area: {box_area_avg}")
        print(f"mean detected mask area: {mask_area_avg}")
    if args.date in ["2023-10-30", "2023-10-31", "2023-11-01", "2023-11-02"]:
        return
    
    # go over each one
    for job_name in social_job_list:
        print(f"text prompt: {job_name}")
        box_area_results, mask_area_results = [], []
        for cfg_w in w_lst:
            box_area, mask_area = [], []
            img_dataset = SimpleDataset(root=f"{args.master_folder}/guide_w{cfg_w}", subset=job_name, exp_date=args.date)
            img_dataload = DataLoader(img_dataset, batch_size=1, shuffle=False, num_workers=4) # per-image inference
            for sample, path in iter(img_dataload):
                with torch.no_grad():
                    new_box_area, new_mask_area = compute_area(path[0], predictor)
                box_area.append(new_box_area)
                mask_area.append(new_mask_area)
            print(f"total number of samples: {len(box_area)}")
            box_area_avg = np.mean(np.array(box_area, dtype=np.float64))
            mask_area_avg = np.mean(np.array(mask_area, dtype=np.float64))
            box_area_results.append(box_area_avg)
            mask_area_results.append(mask_area_avg)
        print(f"detected box area: {box_area_results}")
        print(f"detected mask area: {mask_area_results}")
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--master-folder", type=str, help="path to master folder, not including cfg_w")
    parser.add_argument("--subset-name", type=str, help="string name of a subset to evaluate")
    parser.add_argument("--date", type=str, help="date of experiments for logging")

    args = parser.parse_args()
    print(f"command line arguments: {args}")
    compute_area_metrics(args)