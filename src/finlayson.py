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
    images = p.readimgs("./imgs/natephonePen")

#    for key in images: 
#        image = images[key]
#        height, width = int(image.shape[0] / 2), int(image.shape[1] / 2)
#        image = cv2.resize(image, dsize = (height, width))
#        images[key] = image
#        print(f"Image shape: {image.shape}")

#    print(f"Calculating chromas for {key}...")
    chromas = p.calcImgChromaticity(images)
#    print(f"Removing duplicate chromas for {key}...")

    unique_chromas = {}
    for key in chromas:
        rows = chromas[key].shape[0]
        cols = chromas[key].shape[1]
        print(f"Chromas {key} shape: {chromas[key].shape}")
        unique_chromas[key] = chromas[key].reshape((rows*cols,2))

#    unique_chromas = p.removeDuplicateChromaticities(chromas)

    gray_chromas = {}
    for key in unique_chromas:
        uniqChromas = unique_chromas[key]
        c.chartChromaticity(uniqChromas, False, True, f"./out/{key}_chromas.png")
        entropies = []
        projectionsXY = np.zeros(uniqChromas.shape)
        projectionsByAngle = []
        print(f"Computing entropies for {key}...")
        for theta in range(180):
            projections = p.project(theta, uniqChromas)
            projectionsXY.transpose()[0] = projections
            e = p.entropy(projections)
            entropies.append([theta,e])
            projectionsByAngle.append([theta, projectionsXY])

        entropiesNp = np.array(entropies)
        minEntropyIdx = np.argmin(entropiesNp[:, 1])
        minEntropyTheta = entropiesNp[minEntropyIdx,0]

        # compute projected value for every pixel in chroma image
        rows = chromas[key].shape[0]
        cols = chromas[key].shape[1]
        print(f"Chroma Shape: {chromas[key].shape}, Rows: {rows}, Cols: {cols}")
        flattened = np.reshape(chromas[key], (rows*cols, 2)) 
        flat_gray = p.project(minEntropyTheta,flattened)
        flat_gray = np.exp(flat_gray)
        fg_min = flat_gray.min()
        fg_max = flat_gray.max()
        print(f"max = {fg_max}, min = {fg_min}")
        diff = abs(fg_max - fg_min)
        flat_gray = flat_gray - fg_min
        flat_gray = flat_gray * (1.0 / diff)
        fg_min = flat_gray.min()
        fg_max = flat_gray.max()
        print(f"max = {fg_max}, min = {fg_min}")
        fg_int = flat_gray.astype(int)
        un, cts = np.unique(fg_int, return_counts=True)
        print(np.asarray((un,cts)).T)
        gray_chromas[key] = np.reshape(flat_gray,(rows, cols))
#gray_chromas[key] = np.reshape(fg_int,(rows, cols))
        print(f"Grayscale Shape: {gray_chromas[key].shape}")
        print(f"gray values: {gray_chromas[key]}")
        cv2.imshow("grayscale",gray_chromas[key])
        cv2.waitKey(0)

        minEntropy = entropiesNp[minEntropyIdx][1]
        minEntProjections = projectionsByAngle[minEntropyIdx][1]

        rotated = p.rotate(-minEntropyTheta, minEntProjections)
        minEntProjectionsXY = rotated.transpose()

        c.chartOrigAndProjChromas(unique_chromas[key], minEntProjectionsXY, show = False, save=True, savefn=f"./out/{key}_origAndProjChromas.png")
        
        print(f"{key}: Min Entropy: {minEntropy} Min Entropy Theta: {minEntropyTheta}\n\n")
        c.chartEntropy(entropiesNp, show=False, save=True, savefn=f"./out/{key}_entropy.png")

if __name__ == '__main__':
    main()
