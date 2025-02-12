import cv2
import numpy as np

def detect_letters(image):
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_size = 20

    boxes = [
        (x, y, x + w, y + h)
        for contour in contours
        for x, y, w, h in [cv2.boundingRect(contour)]
        if w > min_size and h > min_size
    ]
    
    return sorted(boxes, key=lambda box: (box[0], box[1]))

def merge_nearby_boxes(boxes, max_distance=30):
    if not boxes:
        return []

    merged = []
    current_box = list(boxes[0])
    
    for box in boxes[1:]:
        x_overlap = (min(current_box[2], box[2]) - max(current_box[0], box[0])) > 0
        y_distance = abs(box[1] - current_box[3])
        
        if x_overlap and y_distance <= max_distance:
            current_box = [
                min(current_box[0], box[0]),
                min(current_box[1], box[1]), 
                max(current_box[2], box[2]),
                max(current_box[3], box[3])
            ]
        else:
            merged.append(tuple(current_box))
            current_box = list(box)
    
    merged.append(tuple(current_box))
    return merged

def pad_and_center_image(img):
    h, w = img.shape
    
    # Make image square
    if h > w:
        pad_left = (h - w) // 2
        pad_right = h - w - pad_left
        padded = cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right, 
                                  cv2.BORDER_CONSTANT, value=255)
    elif w > h:
        pad_top = (w - h) // 2
        pad_bottom = w - h - pad_top
        padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0, 
                                  cv2.BORDER_CONSTANT, value=255)
    else:
        padded = img
    
    # Add uniform padding
    padding = 15
    padded = cv2.copyMakeBorder(padded, padding, padding, padding, padding,
                              cv2.BORDER_CONSTANT, value=255)
    
    # Rotate 90 degrees clockwise and flip horizontally to match training data
    padded = cv2.rotate(padded, cv2.ROTATE_90_CLOCKWISE)
    padded = cv2.flip(padded, 1)
    
    return padded


