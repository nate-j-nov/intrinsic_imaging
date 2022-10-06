# Brenden Collins // Nate Novak
# CS 7180: Advanced Perception
# Fall 2022 
# Library of charting functions

from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
import numpy as np
import process

def chartChromaticity(imgName, chromaDict, show = True, save = False, savefn = "./out/chromaplt.png"): 
    '''
    Function to chart the chromaticity space of an image
    @param imgName: name of image
    @param chromaDict: dictionary of image names and their chromaticity {image name : np array of (G/R, B/R) chromaticity }
    @param show: bool to indicate to show the plot or not
    @param save: bool to indicate to save
    @param savefn: file name to save the image
    '''
    img = chromaDict[imgName]
    xVals = img[:, :, 0]
    yVals = img[:, :, 1]

    plt.scatter(x=xVals, y = yVals, s= 10, facecolors = "none", edgecolors="black")
    plt.xlabel("log(G/R)")
    plt.ylabel("log(B/R)")

    if(show):
        plt.show(); 
    
    if(save): 
        plt.savefig(savefn)


def main(): 
    images = process.readimgs("./imgs/")
    chromas = process.calcImgChromaticity(images)
    keys = list(chromas.keys())
    chartChromaticity(keys[0], chromas)


if __name__ == "__main__": 
    main()
