# Brenden Collins // Nate Novak
# CS 7180: Advanced perception 
# Program to conduct Finlayson Intrinisic imaging algorithm

import cv2 as cv2

def test():
    print("hello world")
    img = cv2.imread("./images/pic.0001.jpg", 1)
    print(img)
    cv2.imshow("test", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def main(): 
    test()

if __name__ == '__main__':
    main()
