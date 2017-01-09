#!/usr/bin/python
import os, csv
import PIL
from PIL import Image
from numpy import genfromtxt
import pandas as pd
import random

path = "/Users/bsoper/Dropbox/Senior Year/Fall/Machine Learning/Project/liscencePlateClassification/images/"
newPath = "/Users/bsoper/Dropbox/Senior Year/Fall/Machine Learning/Project/test/images/"
csvFile = "/Users/bsoper/Dropbox/Senior Year/Fall/Machine Learning/Project/liscencePlateClassification/LiscencePlateBoxes.csv"
outPutSize = (400,400)
orgionalSize = 100.0
NUM_CROPS = 4


boxes=pd.read_csv(csvFile, sep=',',header=None)
dirs = os.listdir( path )
print boxes

def scaleNumbers(w,h,x,y):
    factor =  outPutSize[0]/orgionalSize
    return (int(factor*x), int(factor*y), int(factor*w+(factor*x)), int(factor*h+(factor*y)))

def seccadeImage(fileName,w,h,x,y):
    # Don't seccade if plate is too large in image.
    if w > 90.0 or h > 90.0:
        return
    for suffix in range(0, NUM_CROPS):
        y_top = random.uniform(0, float(y) / 2)
        y_bottom = random.uniform((100 + float(y + h)) / 2, 100)
        x_left = random.uniform(0, float(x) / 2)
        x_right = random.uniform((100 + float(x + w)) / 2, 100)
       
        img = PIL.Image.open(path+fileName)
        img = img.resize(outPutSize, PIL.Image.ANTIALIAS)

        factor =  outPutSize[0]/orgionalSize
        img = img.crop((int(factor*x_left), int(factor*y_top), int(factor*x_right), int(factor*y_bottom)))

        # Resize x, y, w, h
        x_scaled = int((x - x_left) * (100. / (x_right - x_left)))
        y_scaled = int((y - y_top) * (100. / (y_bottom - y_top)))
        w_scaled = int(w * (100. / (x_right - x_left)))
        h_scaled = int(h * (100. / (y_bottom - y_top)))

        newFileName = os.path.splitext(fileName)[0] + '_' + str(suffix) + os.path.splitext(fileName)[1]
        with open(csvFile,'a') as f:
            writer = csv.writer(f)
            writer.writerow([newFileName, w_scaled, h_scaled, x_scaled, y_scaled])
        
        img = img.resize(outPutSize, PIL.Image.ANTIALIAS)

        img.save(path + newFileName)
    

for root, dirs, files in os.walk(path):
    for fileName in files:
        for i in range(len(boxes)):
            if fileName == boxes[0][i]:
               seccadeImage(fileName,boxes[1][i],boxes[2][i],boxes[3][i],boxes[4][i])
