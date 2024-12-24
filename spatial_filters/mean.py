import argparse
from utils.utils import openRGB, showRGB, filters, conv2D, compareToOriginal
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str,
                    help='path to the input image')
parser.add_argument('--kernel_size', type=int, default=3,
                    help='size of the filter')

parser.add_argument('--save', action='store_true')
parser.add_argument('--compare', action='store_true')


args = parser.parse_args()

if __name__ == "__main__":
    print(args.input)
    # print(utils.__dict__)
    img = openRGB(args.input)
    kernel = filters.generateKernel('mean', 3)
    # print(img[0].shape)
    result = np.zeros(img.shape, dtype=np.uint8)

    # currently implementede for 1 color channel only
    result[:,:,0] = conv2D(img[:,:,0], kernel)
    result[:,:,1] = conv2D(img[:,:,1], kernel)
    result[:,:,2] = conv2D(img[:,:,2], kernel)

    if args.save and args.compare:
        compareToOriginal(img, result, fig_title="Mean Filter", filename="mean_result.jpg")
    else:
        showRGB(result)

