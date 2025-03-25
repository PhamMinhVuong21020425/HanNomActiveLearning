import os
import sys
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

from classify.hyperparams import *
from classify.utils import load_model
from classify.transform import ImageTransform
from classify.models import EfficientNetB7_Dropout

yolov5_path = Path('./yolov5')
if str(yolov5_path) not in sys.path:
    sys.path.append(str(yolov5_path))

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device, time_sync

# transform image
resize = 64
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = ImageTransform(resize, mean, std)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prepare network classify
model_classify = EfficientNetB7_Dropout(num_classes=2139)

# prepare model classify
model_classify = load_model(model_classify, f'{WEIGHT_DIR}/efficientnet_b7_last.pth')
model_classify.to(device).eval()

# load model detect
model_detect = attempt_load(weights=f'{WEIGHT_DIR}/yolov5m_best.pt', map_location=device)

# read json file
with open(f'{DATA_DIR}/map/mapping.json', mode='r', encoding='utf-8') as file:
    class_index = json.load(file)


class Predictor():
    def __init__(self):
        self.clas_index = class_index
        self.model_classify = model_classify
        self.model_detect = model_detect

    def preprocess_image(self, img):
        """Preprocess a image for classification."""
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert OpenCV BGR into RGB
        elif isinstance(img, str):
            img = Image.open(img).convert('RGB')
        
        return transform(img, phase='val').unsqueeze(0) # (C, H, W) -> (1, C, H, W)

    def classify_single(self, img):
        # Load and preprocess the input image
        img_tensor = self.preprocess_image(img).to(device)

        # Make a prediction
        with torch.no_grad():
            output = self.model_classify(img_tensor)

        # Calculate probabilities using softmax
        probabilities = F.softmax(output, dim=1)
        prob, predict = torch.max(probabilities, 1) # torch.max return tuple (values, indices)

        class_id = str(predict.item())
        predicted_label = self.clas_index[class_id]

        return class_id, predicted_label, prob.item()
    
    def classify_batch(self, imgs):
        """Classify a batch of cropped images at once"""
        if not imgs:
            return []
        
        batch_tensors = []
        for img in imgs:
            img_tensor = self.preprocess_image(img)
            batch_tensors.append(img_tensor)
        
        # Stack all images into a batch
        batch = torch.cat(batch_tensors).to(device)
        
        # Process all images in one forward pass
        with torch.no_grad():
            outputs = self.model_classify(batch)
        
        # Calculate probabilities using softmax
        probabilities = F.softmax(outputs, dim=1)
        probs, predicts = torch.max(probabilities, 1) # torch.max return tuple (values, indices)
        
        results = []
        for i in range(len(imgs)):
            class_id = str(predicts[i].item())
            predicted_label = self.clas_index[class_id]
            results.append((class_id, predicted_label, probs[i].item()))
        
        return results
    
    def predict(
        self,   
        source='yolov5/data/images',       # image source
        classify=True,                     # classify detected objects
        img_size=640,                      # inference size (pixels)
        conf_thres=0.35,                   # confidence threshold
        iou_thres=0.25,                    # NMS IOU threshold
        max_det=1000,                      # maximum detections per image
        device='',                         # cuda device or cpu
        save_img=False,                    # save annotated images
        save_txt=False,                    # save results to *.txt
        classes=None,                      # filter by class
        augment=False,                     # augmented inference
        line_thickness=2,                  # bounding box thickness
        hide_labels=True,                  # hide labels
        hide_conf=False,                   # hide confidences
        output_dir='runs/predict',         # output directory
        batch_size=32                      # batch size for classification
    ):
        device = select_device(device)
        half = device.type != 'cpu'  # half precision only on CUDA
        
        # Load model
        model = self.model_detect
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # class names
        
        if half:
            model.half()  # to FP16
        
        # Check image size
        img_size = check_img_size(img_size, s=stride)
        
        # Set up output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set Dataloader
        dataset = LoadImages(source, img_size=img_size, stride=stride, auto=True)
        
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, *[img_size, img_size]).to(device).type_as(next(model.parameters())))  # warmup
        
        results = []
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        for path, img, im0s, vid_cap in dataset:
            t1 = time_sync()
            img_copy = im0s.copy()
            # Preprocessing
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            
            t2 = time_sync()
            dt[0] += t2 - t1
            
            # Inference
            pred = model(img, augment=augment)[0]
            t3 = time_sync()
            dt[1] += t3 - t2
            
            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic=False, max_det=max_det)
            dt[2] += time_sync() - t3
            
            # Process detections
            for _, det in enumerate(pred):  # per image
                seen += 1
                p, s, im0 = path, '', im0s.copy()
                p = Path(p)
                save_path = str(output_dir / p.name)
                txt_path = str(output_dir / 'labels' / p.stem) + '.txt'
                s += '%gx%g ' % img.shape[2:]
                
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Batch classification
                    t_class_start = time_sync()
                    cropped_images = []
                    detection_info = []
                    
                    for *xyxy, conf, cls in reversed(det):
                        # Convert tensor to list of integers
                        x1, y1, x2, y2 = [int(coord.item()) for coord in xyxy]
                        
                        img_crop = img_copy[y1:y2, x1:x2]
                        
                        # Skip images that are too small to classify
                        if img_crop.size == 0 or img_crop.shape[0] == 0 or img_crop.shape[1] == 0:
                            continue
                            
                        cropped_images.append(img_crop)
                        # Store detection info including the tensor xyxy for annotation
                        detection_info.append({
                            'xyxy': xyxy,  # Original tensor for annotation
                            'box': [x1, y1, x2, y2],  # Integer coordinates
                            'conf': conf.item(),
                            'cls': cls.item()
                        })
                    
                    # Batch classify all cropped images
                    batch_results = []
                    if classify and cropped_images:
                        # Process in batches to avoid memory issues
                        for i in range(0, len(cropped_images), batch_size):
                            batch = cropped_images[i:i+batch_size]
                            batch_results.extend(self.classify_batch(batch))
                    
                    t_class_end = time_sync()
                    dt[3] += t_class_end - t_class_start
                    
                    # Process results after classification
                    for i, det_info in enumerate(detection_info):
                        if i >= len(batch_results) and classify:
                            continue
                        
                        xyxy = det_info['xyxy']  # Original tensor
                        x1, y1, x2, y2 = det_info['box']  # Integer coordinates
                        conf = det_info['conf']
                        cls = det_info['cls']
                        
                        c = int(cls)
                        detection = {
                            'class': c,
                            'name': names[c],  # Class name
                            'confidence': float(conf),  # Confidence score
                        }

                        if classify and i < len(batch_results):
                            class_id, predicted_label, prob = batch_results[i]
                            detection['class'] = class_id
                            detection['name'] = predicted_label
                            detection['confidence'] = (detection['confidence'] * 2 + prob) / 3

                        detection['coordinates'] = [
                            {'x': x1, 'y': y1},
                            {'x': x1, 'y': y2},
                            {'x': x2, 'y': y2},
                            {'x': x2, 'y': y1},
                            {'x': x1, 'y': y1},
                        ]

                        if detection['confidence'] > 0.25:
                            results.append(detection)
                        
                        # Optional: Write to file
                        if save_txt:
                            # Make sure the labels directory exists
                            os.makedirs(os.path.dirname(txt_path), exist_ok=True)
                            # Convert bbox to xywh format and save
                            xywh = ((x1 + x2) / 2, (y1 + y2) / 2,  # x center, y center
                                  x2 - x1, y2 - y1)  # width, height
                            with open(txt_path, 'a') as f:
                                f.write(f"{c} {' '.join(f'{x:.6f}' for x in xywh)}\n")
                        
                        if save_img:
                            # Add bbox to image
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                
                # Print time (inference-only)
                print(f'{s}Done. ({t3 - t2:.3f}s)')
                
                # Save annotated image
                if save_img:
                    cv2.imwrite(save_path, annotator.result())
        
        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms classify per image at shape {(1, 3, *[img_size, img_size])}' % t)
        if save_txt or save_img:
            print(f"Results saved to {output_dir}")

        return results


if __name__ == '__main__':
    # Predict on an image
    predictor = Predictor()
    source = './yolov5/data/images/test4.png'
    start = time_sync()
    predictor.predict(source=source, classify=False, save_img=True)
    end = time_sync()
    print(f"Total time: {end - start:.3f}s")
