#!/usr/bin/python
import os, csv
import PIL
from PIL import Image
from numpy import genfromtxt
import pandas as pd

path = "/Users/chris/Desktop/plates_br-master/images/"
newPath = "/Users/chris/Desktop/croppedPlates/"
csvFile = "/Users/chris/Desktop/lisencePlates/LiscencePlateBoxes.csv"
outPutSize = (400,400)
orgionalSize = 100.0


boxes=pd.read_csv(csvFile, sep=',',header=None)
dirs = os.listdir( path )
print boxes

def scaleNumbers(w,h,x,y):
    factor =  outPutSize[0]/orgionalSize
    return (int(factor*x), int(factor*y), int(factor*w+(factor*x)), int(factor*h+(factor*y)))

def cropImage(fileName,w,h,x,y):
    img = PIL.Image.open(path+fileName)
    img = img.resize(outPutSize, PIL.Image.ANTIALIAS)
    img = img.crop(scaleNumbers(w,h,x,y))
    img.save(newPath+fileName)

for root, dirs, files in os.walk(path):
    for fileName in files:
        for i in range(len(boxes)):
            if fileName == boxes[0][i]:
                cropImage(fileName,boxes[1][i],boxes[2][i],boxes[3][i],boxes[4][i])
