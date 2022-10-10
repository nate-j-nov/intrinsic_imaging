# Brenden Collins // Nate Novak
# CS 7180: Advanced perception 
# Program to conduct Finlayson Intrinisic imaging algorithm

import numpy as np
import matplotlib.pyplot as plt
import process as p
import charting as c
from decimal import Decimal
import cv2

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
    zeros = np.zeros(pic1.shape)
    print(f"Computing entropies...")
    for theta in range(180):
        projections = p.project(theta, pic1)
        zeros.transpose()[0] = projections
       # if theta % 5 == 0:
       #     plt.xlim(-4.5,4.5)
       #     plt.scatter(zeros[:,0], zeros[:,1], 0.5)
       #     plt.show()
       #     plt.close()
        e = p.entropy(projections)
        entropies.append([theta,e])

    entropy = np.array(entropies)
    minEntropyIdx = np.argmin(entropy[:, 1])
    minEntropyTheta = entropy[minEntropyIdx][0]
    minEntropy = entropy[minEntropyIdx][1]
    
    print(f"Min Entropy: {minEntropy}\nMin Entropy Theta: {minEntropyTheta}")
    c.chartEntropy(entropy, save=True)



if __name__ == '__main__':
    main()
