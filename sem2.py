from PIL import Image
from PIL import ImageFilter
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def nonmaxsup(pixels, size):
	for i in range(1, size[0]-1):
		for j in range(1, size[1]-1):
			if (pixels[i, j+1] >= pixels[i, j] or
				pixels[i, j-1] >= pixels[i, j] or
				pixels[i+1, j] >= pixels[i, j] or
				pixels[i-1, j] >= pixels[i, j] or
				pixels[i+1, j+1] >= pixels[i, j] or
				pixels[i+1, j-1] >= pixels[i, j] or
				pixels[i-1, j+1] >= pixels[i, j] or
				pixels[i-1, j-1] >= pixels[i, j]):
				pixels[i, j] = 0

def derivative(pixels, size):
	for i in range(0, size[0]-1):
		for j in range(0, size[1]-1):
			dx = pixels[i, j] - pixels[i+1, j]
			dy = pixels[i, j] - pixels[i, j+1]
			pixels[i, j] = int(math.sqrt(dx**2 + dy**2))

def threshold(pixels, size, threshold = 127):
	for i in range(0, size[0]-1):
		for j in range(0, size[1]-1):
			if(pixels[i, j] < threshold):
				pixels[i, j] = 0
			else:
				pixels[i, j] = 255

def rgb2gray(processedPixels, originalPixels, size):
	for i in range(0, size[0]-1):
		for j in range(0, size[1]-1):
			processedPixels[i, j] = int((originalPixels[i, j][0] + originalPixels[i, j][1] + originalPixels[i, j][2])/3)

def plotImage(image, subplot):
	plt.figure(0)
	plt.subplot(2, 3, subplot).imshow(np.asarray(image), cmap="gray")

def processImage(imagename, debug = False, filterRadius=0, imageThreshold=5):

	original = Image.open(imagename)
	originalPixels = original.load()
	if debug:
		plotImage(original, 1)
	
	originalSmooth = original.filter(ImageFilter.GaussianBlur(radius=filterRadius))
	originalSmoothPixels = originalSmooth.load();
	if debug:
		plotImage(originalSmooth, 2)

	processed = Image.new("P", original.size)
	processedPixels = processed.load()

	rgb2gray(processedPixels, originalSmoothPixels, original.size)
	if debug:
		plotImage(processed, 3)
	derivative(processedPixels, processed.size)
	if debug:
		plotImage(processed, 4)
	nonmaxsup(processedPixels, processed.size)
	if debug:
		plotImage(processed, 5)
	threshold(processedPixels, processed.size, threshold = imageThreshold)
	if debug:
		plotImage(processed, 6)
		plt.show()
	return processed

images = ["1.tiff", "2.tiff", "3.tiff", "4.tiff", "5.tiff", "6.tiff", "7.tiff"]
for im in images:
	print "Processing"
	processImage(im, debug=True)
	print "Processed"











