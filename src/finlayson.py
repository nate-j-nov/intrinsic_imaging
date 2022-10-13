# Brenden Collins // Nate Novak
# CS 7180: Advanced perception 
# Program to conduct Finlayson Intrinisic imaging algorithm

import numpy as np
import matplotlib.pyplot as plt
import process as p
import charting as c
import cv2
import math 

OFFSET = 0.15

def main(): 
    print(f"Reading images...")
    images = p.readimgs("./imgs")

    print(f"Calculating chromas...")
    chromas = p.calcImgGeoMeanChroma(images)

    unique_chromas = {}
    for key in chromas:
        rows = chromas[key].shape[0]
        cols = chromas[key].shape[1]
        unique_chromas[key] = chromas[key].reshape((rows*cols,2))

    gray_chromas = {}
    for key in unique_chromas:
        uniqChromas = unique_chromas[key]
        c.chartChromaticity(uniqChromas, False, True, f"./out/{key}_chromas.png")
        entropies = []
        projectionsXY = np.zeros(uniqChromas.shape)
        projectionsByAngle = []
        print(f"Computing entropies for {key}...")
        # Project chromaticities according to each angle theta
        for theta in range(180):
            projections = p.project(theta, uniqChromas)
            projectionsXY.transpose()[0] = projections
            e = p.entropy(projections)
            entropies.append([theta,e])
            projectionsByAngle.append([theta, projectionsXY])

        entropiesNp = np.array(entropies)
        minEntropyIdx = np.argmin(entropiesNp[:, 1])
        minEntropyTheta = entropiesNp[minEntropyIdx,0]
        minEntropy = entropiesNp[minEntropyIdx][1]
        minEntProjections = projectionsByAngle[minEntropyIdx][1]

        print(f"{key}: Min Entropy: {minEntropy} Min Entropy Theta: {minEntropyTheta}")

        # compute projected value for every pixel in chroma image
        print(f"Generating grayscale image for {key}...")
        rows = chromas[key].shape[0]
        cols = chromas[key].shape[1]
        flattened = np.reshape(chromas[key], (rows*cols, 2)) 
        flat_gray = p.project(minEntropyTheta,flattened)
        flat_gray = np.exp(flat_gray)
        fg_min = flat_gray.min()
        fg_max = flat_gray.max()
        print(f"max = {fg_max}, min = {fg_min}")
        diff = abs(fg_max - fg_min)
        flat_gray = flat_gray - fg_min
        flat_gray = flat_gray * ((1.0) / diff)
        fg_min = flat_gray.min()
        fg_max = flat_gray.max()
        print(f"1st adjustment max = {fg_max}, min = {fg_min}")
        flat_gray = flat_gray * (1.0 - OFFSET)
        flat_gray = flat_gray + OFFSET 
        fg_min = flat_gray.min()
        fg_max = flat_gray.max()
        print(f"2nd adjustment max = {fg_max}, min = {fg_min}")

        gray_chromas[key] = np.reshape(flat_gray,(rows, cols))
        cv2.imshow("grayscale",gray_chromas[key])
        cv2.imwrite(f"./out/{key}_grayscale.png", gray_chromas[key]*255)
        cv2.waitKey(0)

        rotated = p.rotate(-minEntropyTheta, minEntProjections)
        minEntProjectionsXY = rotated.transpose()

        c.chartOrigAndProjChromas(unique_chromas[key], minEntProjectionsXY, show = False, save=True, savefn=f"./out/{key}_origAndProjChromas.png")
        
        c.chartEntropy(entropiesNp, show=False, save=True, savefn=f"./out/{key}_entropy.png")
        print("\n")

if __name__ == '__main__':
    main()
