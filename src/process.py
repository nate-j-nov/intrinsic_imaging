# Brenden Collins // Nate Novak
# CS 7180: Advanced perception 
# Module to hold all of the steps of the algorithm

from gc import collect
from genericpath import isfile
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
import os
import math

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
    @return: dictionary of image names and their chromaticity 
        {image name : np array of (log(G/R), log(B/R)) chromaticity }
    '''
    chromaDict = {}
    for key in imagesDict: 
        bgrImage = imagesDict[key]
        height, width, depth = bgrImage.shape
        chroma = np.zeros((height, width, 2), dtype=float)
        for row in range(height): 
            for col in range(width): 
                pixel = bgrImage[row, col]
                # add 1 to avoid divide by zero and log(0) errors
                b, g, r = pixel[0]+1, pixel[1]+1, pixel[2]+1 

                chroma[row, col, 0] = math.log(g / r)
                chroma[row, col, 1] = math.log(b / r)
        
        chromaDict[key] = chroma

    return chromaDict


def removeDuplicateChromaticities(chromaDict): 
    '''
    Function to remove duplicate chromaticities
    @param chromaDict: dictionary of image names and their chromaticity {image name : np array of (G/R, B/R) chromaticity }
    @return: dictionary of image names and unique chromaticities {image name : nparray of (G/R, B/R) chromaticity }
    '''
    uniqueChromasDict = {}
    for key in chromaDict: 
        chroma = chromaDict[key]
        width, height, depth = chroma.shape
        
        uniqueChromas = set()

        for i in range(width): 
            for j in range(height): 
                pixelchr = chroma[i][j]
                gr = pixelchr[0]
                br = pixelchr[1]
                uniqueChromas.add(tuple([gr, br]))
        
        print(f"chroma len: {chroma.shape}\nuniquechromas len: {len(uniqueChromas)}")
        chromalist = []

        for item in uniqueChromas: 
            chromalist.append([item[0], item[1]])

        uniqueChromasDict[key] = np.array(chromalist)

    return uniqueChromasDict

#def contains(collection, comparer):
#    '''
#    Function to determine if a collection contains a certain item 
#    @param collection: nparray to determine to check if an item is contained within it
#    @param comparer: item to dtermine if it exists in it
#    @return: boolean indicating if comparer is in the collection
#    ''' 
#    for item in collection: 
#        if math.isclose(item[0], comparer[0]) and math.isclose(item[1], comparer[1]): 
#            return True
#    
#    return False;
#
#    pass

def project(theta: int, chromas: np.ndarray) -> np.ndarray:
    ''' Function to project 2d points according to angle theta
    @param theta: integer representing the angle of rotation in degrees
    @param chromas: Nx2 ndarray of chromaticity values
    @return: Nx1 ndarray of projected chromaticity intensities
    '''
    assert chromas.shape[0] == 2 or chromas.shape[1] == 2, "Dimension of ndarray is incorrect"
    # if ndarray is Nx2, transpose it
    if chromas.shape[1] == 2:
        chromas = chromas.transpose()

    # create rotation matrix
    theta_r = math.radians(theta)
    cosa = math.cos(theta_r)
    sina = math.sin(theta_r)
    rotation = np.array([[cosa, -sina], [sina, cosa]])
    print(rotation) # print rotation matrix

    # matrix multiply rotation by chromas 
    rotated = np.matmul(rotation, chromas)
    print(rotated)

    projected = rotated[0]
    print(f"projected = {projected}")
    return projected

    


def main(): 
    images = readimgs("./imgs/")
    chromas = calcImgChromaticity(images)
    unique_chromas = removeDuplicateChromaticities(chromas)
    for key in unique_chromas.keys():
        print(f"unique_chromas key = {key}")
    pic1 = unique_chromas['./imgs/pic.0001']
    print(f"pic1 shape: {pic1.shape}")
    zeros = np.zeros(pic1.shape)
    print(f"zeros shape: {zeros.shape}")
    for i in range(12):
        theta = i*15
        projections = project(theta, pic1)
        zeros.transpose()[0] = projections
        print(f"zero_proj = {zeros}")
        plt.scatter(zeros[:,0], zeros[:,1], 1)
        plt.show()
        


if __name__ == "__main__":
    main()
