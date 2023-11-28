import cv2


def adjust_contrast(image, alpha, beta):
 
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image


 
image_path = 'Photos/image.jpg'  
image = cv2.imread(image_path)

 
if image is not None:
 
    adjusted_image = adjust_contrast(image, alpha=2.0, beta=0)

  
    cv2.imshow('Original Image', image)
    cv2.imshow('Adjusted Image', adjusted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Unable to load the image.")
