import math
import numpy as np
from process import project

def main():
# a = np.array([[1,2,3,4],[5,6,7,8],[8,10,11,12]])
    a = np.array([[1,2,3],[1,1,5]])
    print(f"Test array shape: \n{a.shape}")
    print(f"Test array: \n{a}")

    alpha = 40

    output = project(alpha, a)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")



if __name__ == "__main__":
    main()
