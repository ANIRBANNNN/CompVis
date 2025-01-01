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
    img = openRGB(args.input)[:,:,0]
    # print(img[0,0])
    
    # showRGB(img)
    f = filters()
    
    
    result = np.zeros_like(img)

    # Applying median filter to each color channel independently
    result = f.generateKernel(kernel_type="median",kernel_size=args.kernel_size,image=img)
    # result[:,:,1] = f.generateKernel(kernel_type="Median",kernel_size=args.kernel_size,image=img[:,:,1])
    # result[:,:,2] = f.generateKernel(kernel_type="Median",kernel_size=args.kernel_size,image=img[:,:,2])
    
    

    if args.save and args.compare:
        compareToOriginal(img, result, fig_title="Median Filter", filename="med_result.jpg")
    else:
        # print(result)
        showRGB(result)