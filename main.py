import cv2
from ultralytics import YOLO
import numpy as np


class YoloDetector:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model)
        self.classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
            8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
            22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
            29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
            48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
            55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
            62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
            76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
        }

    def run(self, source):
        return self.model.predict(source, classes=[67, 39], conf=0.8)[0]


def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Calculate area of intersection
    area_inter = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

    # Calculate area of individual bounding boxes
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate union area
    area_union = area_box1 + area_box2 - area_inter

    # Calculate IoU
    iou = area_inter / area_union if area_union > 0 else 0.0
    return iou, (int(x1_inter), int(y1_inter), int(x2_inter), int(y2_inter))


if __name__ == '__main__':
    yolo = YoloDetector("asset/yolo/yolov8s_openvino_model")
    cap = cv2.VideoCapture(0)

    while True:
        has_frame, img = cap.read()
        if not has_frame:
            break
        w, h = img.shape[1], img.shape[0]

        # Dictionary counter
        counter = {}

        # Yolov8
        result = yolo.run(img)
        xyxy = result.boxes.xyxy.clone()
        classes = result.boxes.cls.clone()

        custom_box = [int(w / 4), int(h / 4), int(w * (3 / 4)), int(h * (3 / 4))]
        cv2.rectangle(img, (custom_box[0], custom_box[1]), (custom_box[2], custom_box[3]), (255, 255, 255), 2)

        # Filter out the person classes
        non_zero_idx = (classes != 0).nonzero().squeeze()
        length_nonzero = non_zero_idx.numel()  # Use numel() for scalar tensor

        if length_nonzero > 0:
            non_zero_idx = non_zero_idx.tolist() if length_nonzero > 1 else [non_zero_idx.item()]
            for idx in non_zero_idx:
                iou, (xou1, you1, xou2, you2) = calculate_iou(custom_box, xyxy[idx])

                sub_img = img[you1:you2, xou1:xou2]
                white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
                res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
                img[you1:you2, xou1:xou2] = res

                cv2.putText(img, f"{iou:.2f}", (xou1, you2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                if iou > 0:  # The result of IoU is bias,
                    # if the object is small and far, caused of that I change it to 0
                    object_name = yolo.classes[int(classes[idx])]
                    if object_name in counter:
                        counter[object_name] += 1
                    else:
                        counter[object_name] = 1

        print(counter)
        cv2.imshow("webcam", result.plot())
        if cv2.waitKey(20) & 0xFF == ord("q"):  # press q to quit
            break

    cap.release()
    cv2.destroyAllWindows()
