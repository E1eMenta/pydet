import cv2

TEXT_COLOR = (255, 0, 255)

def draw_box(image, bbox, class_id, class_idx_to_name=None, color=(255, 0, 0), thickness=2):
    height, width, _ = image.shape
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max = int(x_min * width), int(x_max * width)
    y_min, y_max = int(y_min * height), int(y_max * height)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    if class_idx_to_name is not None:
        class_name = class_idx_to_name[class_id]
        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(image, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
        cv2.putText(image, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_COLOR,
                    lineType=cv2.LINE_AA)
    return image

def draw_boxes(image, bboxes, class_ids=None, class_idx_to_name=None, scores=None, color=(255, 0, 0), thickness=2):
    height, width, _ = image.shape
    for idx, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = bbox
        x_min, x_max = int(x_min * width), int(x_max * width)
        y_min, y_max = int(y_min * height), int(y_max * height)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
        if class_idx_to_name is not None:
            class_name = class_idx_to_name[class_ids[idx]]
            if scores is not None:
                class_name += " {:.2f}".format(scores[idx])
            ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            cv2.rectangle(image, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
            cv2.putText(image, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, TEXT_COLOR,
                        lineType=cv2.LINE_AA)
    return image