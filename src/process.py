# Brenden Collins // Nate Novak
# CS 7180: Advanced perception 
# Module to hold all of the steps of the algorithm

import cv2
import numpy as np
from matplotlib import pyplot as plt
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
            imgName, ext = split[0], split[1]
            if ext == ".jpg" or ext == ".png": 
                # if we get here, this is 1.) a file, and 2.) the file is an image
                fn = os.path.splitext(file)[0]
                img = cv2.imread(name, 1)
                images[fn] = img
    
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

    
def calcImgGeoMeanChroma(imagesDict):
    '''
    Function to calculate the geometric mean chromaticity of each pixel in the image
    @param imagesDict: dictionary of images {image name : np array}
    @return: dictionary of image names and their 3D geometric mean chromaticity 
        {image name : np array of (log(R/(R*G*B)), log(G/(R*G*B)), log(B/(R*G*B))) chromaticity }
    '''
    chromaDict = {}
    # chi1 and chi2 are an orthonormal basis for the 2d plane
    #   normal to the vector u = (1/srt(3))(1,1,1) - calculated by hand
    chi1 = np.array((1.0/math.sqrt(2), -1.0/math.sqrt(2), 0))
    chi2 = np.array((1.0/math.sqrt(6), 1.0/math.sqrt(6), -math.sqrt(2)/math.sqrt(3)))
    for key in imagesDict: 
        bgrImage = imagesDict[key]
        height, width, depth = bgrImage.shape
        chroma = np.zeros((height, width, 2), dtype=float)
        for row in range(height): 
            for col in range(width): 
                pixel = bgrImage[row, col]
                # add 1 to avoid divide by zero and log(0) errors
                b, g, r = pixel[0]+1, pixel[1]+1, pixel[2]+1 
                M = math.pow(b*g*r*1.0, 1/3) # geometric mean divisor

                # compute rho values with geometric mean divisor 
                rb = math.log(b*1.0 / M)
                rg = math.log(g*1.0 / M)
                rr = math.log(r*1.0 / M)

                # compute 2d chroma space by projecting onto chi1 and chi2
                chroma[row, col, 0] = rb*chi1[0] + rg*chi1[1] + rr*chi1[2]                
                chroma[row, col, 1] = rb*chi2[0] + rg*chi2[1] + rr*chi2[2]                
        
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
        
        chromalist = []

        for item in uniqueChromas: 
            chromalist.append([item[0], item[1]])

        uniqueChromasDict[key] = np.array(chromalist)

    return uniqueChromasDict

def rotate(theta: int, chromas: np.ndarray) -> np.ndarray: 
    '''
    Function to rotate an nd array by angle theta
    @param theta: integer representing the angle of rotation in degrees
    @param chromas: Nx2 ndarray of chromaticity values
    @param chromas: Nx2 ndarray of chromaticity values
    @return: Nx2 ndarray of rotate chromaticit points
    '''
    if chromas.shape[1] == 2:
        chromas = chromas.transpose()

    # create rotation matrix
    theta_r = math.radians(theta)
    cosa = math.cos(theta_r)
    sina = math.sin(theta_r)
    rotation = np.array([[cosa, -sina], [sina, cosa]])

    # matrix multiply rotation by chromas 
    rotated = np.matmul(rotation, chromas)

    return rotated

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

    # matrix multiply rotation by chromas 
    rotated = rotate(theta, chromas)

    projected = rotated[0]
    return projected

def entropy(chromas: np.ndarray) -> float:
    ''' Computes entropy of an ndarray as specified in Finlayson.
        Unlike in Finlayson, we do not excise data below the 5th
        above the 95th percentile, but rather keep the whole array.
        @param chromas: ndarray of 1d chromaticies (after projection)
        @return: entropy as a float
    '''
    hist = np.histogram(chromas, bins=64)[0]
    sum_bins = hist.sum()
    px = hist/sum_bins
    px = np.sort(px)
    px = np.trim_zeros(px)
    px = -px*np.log2(px)
    entropy = px.sum()
    return entropy


def test(): 
    ''' 
    Function to test this library
    '''
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

def main(): 
    test()
        
if __name__ == "__main__":
    main()
