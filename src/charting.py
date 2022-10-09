# Brenden Collins // Nate Novak
# CS 7180: Advanced Perception
# Fall 2022 
# Library of charting functions

from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
import numpy as np
import process
import random as rand

def chartChromaticity(chroma, show = True, save = False, savefn = "./out/chromaplt.png"): 
    '''
    Function to chart the chromaticity space of an image
    @param chroma: np array of (G/R, B/R) chromaticity
    @param show: bool to indicate to show the plot or not
    @param save: bool to indicate to save the plot
    @param savefn: file name to save the image
    '''
    print(chroma.shape)
    xvals = chroma[:, 0]
    yvals = chroma[:, 1]

    plt.scatter(x = xvals, y = yvals, s = 10, facecolors = "none", edgecolors = "black")
    plt.xlabel("log(G/R)")
    plt.ylabel("log(B/R)")

    if(show):
        plt.show() 
    
    if(save): 
        plt.savefig(savefn)

    plt.close()

def chartEntropy(entropyData, show = True, save = False, savefn = "./out/entropyplt.png"): 
    '''
    Function to chart the entropy of an image at each angle of projection @param entropyData: nparray of size with the degree as the x value and entropy as the y value
    @param show: bool to indicate to show the plot or not
    @param save: bool to indicate to save the plot
    @param savefn: file name to save the image
    '''
    xvals = entropyData[:, 0]
    yvals = entropyData[:, 1]

    plt.plot(xvals, yvals)
    plt.plot()
    plt.xlabel("Angle (Degrees)")
    plt.ylabel("Entropy")

    if(show): 
        plt.show()
    
    if(save): 
        plt.savefig(savefn)
    
    plt.close()


def chartTests(): 
    '''
    Function to test charting functions
    '''
    images = process.readimgs("./imgs/")
    chromas = process.calcImgChromaticity(images)
    uniqueChromas = process.removeDuplicateChromaticities(chromas)

    keys = list(chromas.keys())
    chroma = uniqueChromas[keys[0]]
    chartChromaticity(chroma)
    numangles = 180
    testdata = np.zeros((numangles, 2), dtype=float)
    height, width = testdata.shape
    for i in range(height): 
        point = rand.gauss(0.0, 10.0)
        testdata[i][0] = i
        testdata[i][1] = point

    chartEntropy(testdata)

def main(): 
    chartTests()

if __name__ == "__main__": 
    main()
