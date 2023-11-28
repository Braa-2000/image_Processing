import cv2
import numpy as np
def apply_average_filter(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))
image_path = 'Photos/image.jpg'   
image = cv2.imread(image_path)
if image is not None:
    kernel_size = 5   
    filtered_image = apply_average_filter(image, kernel_size)
    cv2.imshow('Original Image', image)
    cv2.imshow('Filtered Image', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Unable to load the image.")
