import argparse
from utils.utils import openRGB, grayscale, isGray
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str,
                    help='path to the input image')

parser.add_argument('--show', action='store_true')
parser.add_argument('--save', action='store_true')
# parser.add_argument('--compare', action='store_true')
parser.add_argument('--output', type=str,
                    help="path to save the file")

parser.add_argument('--percent', type=int, default=50, choices=range(0,101), metavar="[0-100]",
                    help='percentage of the total pixels to be corrupted')

args = parser.parse_args()

if __name__ == "__main__":
    
    img = openRGB(args.input)
    
    gray_img = grayscale(img)

    h, w = gray_img.shape

    cords = set(np.random.choice(h*w, h*w*args.percent//100))
    # print(cords)

    for i in range(h):
        for j in range(w):
            if (i*w + j) in cords:
                gray_img[i,j] = np.random.choice([0,255])

    if args.save:
        if isGray(gray_img):
            plt.imsave(args.output, gray_img, cmap='gray')
        else:
            plt.imsave(args.output, gray_img)

    if args.show:
        print(isGray(gray_img))
        if isGray(gray_img):
            plt.imshow(gray_img, cmap='gray')
        else:
            plt.imshow(gray_img)
        plt.show()
    

