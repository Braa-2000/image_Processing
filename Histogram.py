import cv2
import matplotlib.pyplot as plt

def plot_histogram(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    color_histograms = [cv2.calcHist([image], [i], None, [256], [0, 256]) for i in range(3)]
    gray_histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    colors = ['b', 'g', 'r']
    labels = ['Blue', 'Green', 'Red', 'Grayscale']
    histograms = color_histograms + [gray_histogram]

    for i, histogram in enumerate(histograms):
        plt.plot(histogram, color=colors[i-1])
        plt.xlim([0, 256])

    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.legend(labels)
    plt.show()
image_path = 'Photos/image.jpg'   
image = cv2.imread(image_path)
if image is not None:
    plot_histogram(image)
else:
    print("Error: Unable to load the image.")
