import cv2

def apply_gaussian_filter(image, kernel_size, sigma):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)

 
image_path = 'Photos/image.jpg'   
image = cv2.imread(image_path)

 
if image is not None:
   
    kernel_size = 5   
    sigma = 0   

   
    filtered_image = apply_gaussian_filter(image, kernel_size, sigma)

     
    cv2.imshow('Original Image', image)
    cv2.imshow('Filtered Image', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Unable to load the image.")
