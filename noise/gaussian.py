import argparse
from utils.utils import openRGB, showRGB, filters, conv2D, compareToOriginal
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str,
                    help='path to the input image')

parser.add_argument('--show', action='store_true')
# parser.add_argument('--compare', action='store_true')
parser.add_argument('--output', type=str,
                    help="path to save the file")

parser.add_argument('--mean', type=float, default=0.0,
                    help='mean of the Gaussian dist')

parser.add_argument('--std', type=float, default=50.0,
                    help='standard deviation of the Gaussian dist')

args = parser.parse_args()

if __name__ == "__main__":
    
    img = openRGB(args.input)
    gaussian_noise = np.random.normal(args.mean, args.std, img.shape)

    noisy_img = img + gaussian_noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    plt.imsave(args.output, noisy_img)

    if args.show:
        plt.imshow(noisy_img)
        plt.show()
    

