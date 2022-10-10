# Brenden Collins // Nate Novak
# CS 7180: Advanced perception 
# Program to conduct Finlayson Intrinisic imaging algorithm

import numpy as np
import matplotlib.pyplot as plt
import process as p
import charting as c

def main(): 
    print(f"Reading images...")
    images = p.readimgs("./imgs/")
    print(f"Calculating chromas...")
    chromas = p.calcImgChromaticity(images)
    print(f"Removing duplicate chromas...")
    unique_chromas = p.removeDuplicateChromaticities(chromas)
    for key in unique_chromas.keys():
        print(f"unique_chromas key = {key}")
    pic1 = unique_chromas['./imgs/shady_person']
    print(f"pic1 shape: {pic1.shape}")
    entropies = []
    zeros = np.zeros(pic1.shape)
    print(f"Computing entropies...")
    for theta in range(180):
        projections = p.project(theta, pic1)
        zeros.transpose()[0] = projections
        if theta % 5 == 0:
            plt.xlim(-4.5,4.5)
            plt.scatter(zeros[:,0], zeros[:,1], 0.5)
            plt.show()
            plt.close()
        e = p.entropy(projections)
        entropies.append([theta,e])
    entropy = np.array(entropies)
    c.chartEntropy(entropy, save=True)



if __name__ == '__main__':
    main()
