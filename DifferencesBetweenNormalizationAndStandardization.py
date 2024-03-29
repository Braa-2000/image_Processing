import cv2
import numpy as np
import matplotlib.pyplot as plt
####################################
def normalize_image(image):
    normalized_image = image / 255.0
    return normalized_image
def standardize_image(image):
    standardized_image = (image - np.mean(image)) / np.std(image)
    return standardized_image
image_path = 'Photos/image.jpg'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# normalization and standardization
normalized_image = normalize_image(original_image)
standardized_image = standardize_image(original_image) 
plt.figure(figsize=(10, 5))


################# Original Image ###################
plt.subplot(1, 3, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Normalized Image')

################# Original Image ###################
plt.subplot(1, 3, 2)
plt.imshow(normalized_image, cmap='gray')
plt.title('Normalized Image')

################# Standardized Image ###################
plt.subplot(1, 3, 3)
plt.imshow(standardized_image, cmap='gray')
plt.title('Standardized Image')

plt.show()
