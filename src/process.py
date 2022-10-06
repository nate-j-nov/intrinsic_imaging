# Brenden Collins // Nate Novak
# CS 7180: Advanced perception 
# Module to hold all of the steps of the algorithm

from genericpath import isfile
import cv2
from importlib_metadata import files
import numpy as np
import os

def test():
    '''Function to test dependencies'''
    print("hello world")
    img = cv2.imread("./imgs/pic.0001.jpg", 1)
    print(img)
    cv2.imshow("test", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def readimgs(dir): 
    '''
    Function to read all the images in a directory
    @param dir: indicates the directory to walk
    @return: dict of numpy arrays {image name : np array}
    '''
    images = {}
    for file in os.listdir(dir): 
        name = os.path.join(dir, file)
        if os.path.isfile(name): 
            split = os.path.splitext(name)
            imgName = split[0]
            ext = split[1]
            if ext == ".jpg" or ext == ".png": 
                # if we get here, this is 1.) a file, and 2.) the file is an image
                img = cv2.imread(name, 1)
                images[imgName] = img
    
    return images


def calcImgChromaticity(imagesDict):
    '''
    Function to calculate the chromaticity of each pixel in the image
    @param imagesDict: dictionary of images {image name : np array}
    @return: dictionary of image names and their chromaticity {image name : np array of (G/R, B/R) chromaticity }
    '''
    chromaDict = {}
    for key in imagesDict: 
        bgrImage = imagesDict[key]
        height, width, depth = bgrImage.shape
        chroma = np.zeros((height, width, 2), dtype=float)
        for row in range(height): 
            for col in range(width): 
                pixel = bgrImage[row, col]
                b, g, r = pixel[0], pixel[1], pixel[2]

                if(r == 0): continue

                chroma[row, col, 0] = g / r
                chroma[row, col, 1] = b / r
        
        chromaDict[key] = chroma

    return chromaDict


def main(): 
    images = readimgs("./imgs/")
    chromas = calcImgChromaticity(images)
    print(chromas)
    print("Branching test")

if __name__ == "__main__":
    main()