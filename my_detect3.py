from pathlib import Path

import cv2
import numpy as np
import torch

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import LOGGER, non_max_suppression, scale_coords
from utils.plots import Annotator, colors


def initialize_model(weights, device):
    model = DetectMultiBackend(weights, device=device, dnn=False)
    return model

def initialize_reader(source):
    if source == '0':
        source = int(source)
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
    return cap


def initialize_writer():
    pass


def get_frame(reader):
    for i in range(10):
        reader.read()
    ret, img = reader.read()
    return ret, img

def prepare_frame(frame, imgsz, device):
    im0s = [frame.copy()]
    
    im = [letterbox(x, imgsz, stride=32, auto=True)[0] for x in im0s]

    # Stack
    im = np.stack(im, 0)

    # Convert
    im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device)
    im = im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    return im

def definition_of_predictions(model, im, imgsz):
    model.warmup(imgsz=(1, 3, imgsz), half=False)  # warmup
    # Inference
    pred = model(im, augment=False, visualize=False)

    # NMS
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, max_det=100)
    print(pred)
    return pred


def show_result(model, view_img, source, im, frame, pred):
    names = model.names
    path = source
    im = im
    im0s = [frame.copy()]
    s = ''

    # Process predictions
    for i, det in enumerate(pred):  # per image
            if source == '0':  # batch_size >= 1
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


def main(source, device, weights, view_img, imgsz):
    model = initialize_model(weights, device)
    reader = initialize_reader(source)
    # writer = initialize_writer(...params...) // Этот не очень нужен, но можно оставить.

    while True:
        ret, frame = get_frame(reader)
        if not ret:
            break
        im = prepare_frame(frame, imgsz, device)
        pred = definition_of_predictions(model, im, imgsz)
        show_result(model, view_img, source, im, frame, pred)

if __name__ == "__main__":
    source='0'
    device='cpu'
    weights='yolov5s.pt'
    view_img=True
    imgsz=640 
    main(source, device, weights, view_img, imgsz)