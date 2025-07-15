import cv2
import numpy as np

# Load and preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    return thresh

# Extract contour area
def get_contour_area(thresh_img):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        return area, largest_contour
    return 0, None

# Score cube drawing
def score_cube(image_path, reference_area):
    thresh_img = preprocess_image(image_path)
    area, contour = get_contour_area(thresh_img)

    if contour is not None:
        if reference_area * 0.9 <= area <= reference_area * 1.1:
            return 2  # Correct Cube
        elif reference_area * 0.5 <= area < reference_area * 0.9:
            return 1  # General Cube Shape
    return 0  # Incorrect Cube
