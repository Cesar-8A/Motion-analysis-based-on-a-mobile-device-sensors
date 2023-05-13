######-----Set up environment for yolo use-----######
 #Install required libraries this way

import numpy as np
from ultralytics import YOLO

 #import yolo model x version this way
model = YOLO('yolov8x-seg.pt')
#model.to('cuda') #uncomment this if cuda is available


######-----The callable function for getting segmentated mask-----######
#Segmentation, this function receives a single image per use, the image must be a numpy matrix 
#this function returns a mask of the segmentated person or returns 0 if not persons were found
def yoloseg(img):
  #Get model results
  results = model(img)
  results = results[0]
  clases = results.boxes.cls.cpu().numpy().astype(int)
  #Find person index mask
  person = 0
  index = np.where(clases == person)[0]
  if np.size(index) == 0:
    #No persons were found
    return 0
  #return person mask as numpy array where mask is set to 1
  return results.masks.data.cpu().numpy()[index[0]].astype(int)


######-----Example of function call-----######
import cv2
path='insert your image path here'
image = cv2.imread(path)
res = yoloseg(image)
######-----show the mask-----######
import matplotlib.pyplot as plt
plt.imshow(res)
plt.show()

