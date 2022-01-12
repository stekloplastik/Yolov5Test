from pathlib import Path
import re

import cv2
import torch
import torch.backends.cudnn as cudnn


from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size,
                           non_max_suppression, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import time_sync


def reading_and_preparation(source):
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    webcam = source.isnumeric()
    if is_file:
        source = check_file(source)
        webcam = False
        return source, webcam
    else:
        return source, webcam

def calling_and_parsing(source, imgsz, device, weights):
    source, webcam = reading_and_preparation(source)
    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    return model, dataset, names, webcam

def show_result(source, imgsz, device, weights, view_img):
    model, dataset, names, webcam = calling_and_parsing(source, imgsz, device, weights)

    # Run inference
    model.warmup(imgsz=(1, 3, imgsz), half=False)  # warmup
    print(dataset)
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        print(im)
        # Inference
        pred = model(im, augment=False, visualize=False)

        # NMS
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, max_det=100)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            if webcam:  # batch_size >= 1
                p, im0 = path[i], im0s[i].copy()
                s += f'{i}: '
            else:
                p, im0 = path, im0s.copy()

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string
            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = (f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
            # Print time (inference-only)
            LOGGER.info(f'{s}Done.')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

if __name__ == "__main__":
    weights='yolov5s.pt'
    source='0'
    imgsz=640 
    device='cpu' 
    view_img=True
    show_result(source, imgsz, device, weights, view_img)
