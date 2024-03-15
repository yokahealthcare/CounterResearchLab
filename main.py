import cv2
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model)
        self.classes = self.model.names

    def run(self, source):
        return self.model.predict(source, classes=[39, 67], conf=0.9, augment=True, agnostic_nms=True)[0]


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
    yolo = YoloDetector("asset/yolo/yolov8x.pt")

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
                cv2.rectangle(img, (xou1, you1), (xou2, you2), (255, 255, 0), thickness=-1)
                cv2.putText(img, f"{iou:.2f}", (xou1, you2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

                if iou > 0.3:
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
