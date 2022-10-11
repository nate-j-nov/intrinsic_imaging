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
    images = p.readimgs("./imgs/brendenCampus")

    for key in images: 
        image = images[key]
        height, width = int(image.shape[0] / 2), int(image.shape[1] / 2)
        image = cv2.resize(image, dsize = (height, width))
        images[key] = image

        print(f"Calculating chromas for {key}...")
        chromas = p.calcImgChromaticity(images)
        print(f"Removing duplicate chromas for {key}...")

        unique_chromas = p.removeDuplicateChromaticities(chromas)
        uniqChromas = unique_chromas[key]
        c.chartChromaticity(uniqChromas, False, True, f"./out/{key}_chromas.png")
        entropies = []
        projectionsXY = np.zeros(uniqChromas.shape)
        projectionsByAngle = []
        print(f"Computing entropies for {key}...")
        for theta in range(180):
            projections = p.project(theta, uniqChromas)
            projectionsXY.transpose()[1] = projections
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

        c.chartOrigAndProjChromas(unique_chromas[key], minEntProjectionsXY, show = False, save=True, savefn=f"./out/{key}_origAndProjChromas.png")
        
        print(f"{key}: Min Entropy: {minEntropy} Min Entropy Theta: {minEntropyTheta}\n\n")
        c.chartEntropy(entropiesNp, show=False, save=True, savefn=f"./out/{key}_entropy.png")

if __name__ == '__main__':
    main()
