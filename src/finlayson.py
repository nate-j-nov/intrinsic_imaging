# Brenden Collins // Nate Novak
# CS 7180: Advanced perception 
# Program to conduct Finlayson Intrinisic imaging algorithm

import numpy as np
import matplotlib.pyplot as plt
import process as p
import charting as c
from decimal import Decimal
import cv2
import math 

def main(): 
    print(f"Reading images...")
    images = p.readimgs("./imgs/")

    for key in images: 
        image = images[key]
        height, width = int(image.shape[0] / 2), int(image.shape[1] / 2)
        image = cv2.resize(image, dsize = (height, width))
        images[key] = image

    print(f"Calculating chromas...")
    chromas = p.calcImgChromaticity(images)
    print(f"Removing duplicate chromas...")
    unique_chromas = p.removeDuplicateChromaticities(chromas)
    for key in unique_chromas.keys():
        print(f"unique_chromas key = {key}")
    pic1 = unique_chromas['./imgs/shady_person']
    entropies = []
    projectionsXY = np.zeros(pic1.shape)
    projectionsByAngle = []
    print(f"Computing entropies...")
    for theta in range(180):
        projections = p.project(theta, pic1)
        #print(projections)
        projectionsXY.transpose()[1] = projections
        #print(f"projectionsXY.shape: {projectionsXY.shape}")
        e = p.entropy(projections)
        entropies.append([theta,e])
        projectionsByAngle.append([theta, projectionsXY])

    entropiesNp = np.array(entropies)
    minEntropyIdx = np.argmin(entropiesNp[:, 1])
    minEntropyTheta = entropiesNp[minEntropyIdx][0]
    minEntropy = entropiesNp[minEntropyIdx][1]
    minEntProjections = projectionsByAngle[minEntropyIdx][1]

    rotated = p.rotate(-minEntropyTheta, minEntProjections)
    minEntProjectionsXY = rotated.transpose()

    c.chartOrigAndProjChromas(unique_chromas["./imgs/shady_person"], minEntProjectionsXY)
    
    print(f"Min Entropy: {minEntropy}\nMin Entropy Theta: {minEntropyTheta}")
    c.chartEntropy(entropiesNp, save=True)

if __name__ == '__main__':
    main()
