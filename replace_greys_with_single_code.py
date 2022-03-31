import os
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np
import argparse
import pdb

def recons_image_from_cb_indices(results):
    input_file = results.input_file
    zero_file = results.zero_file
    zero_dict = {}
    count = 0
    with open(zero_file) as fp:
        for line in fp:
            line = line.rstrip("\n")
            zero_dict[line]  = 1
    with open(input_file) as fp:
        pick_count = 0
        val = 0
        for line in fp:
            line = line.rstrip("\n")
            if line in zero_dict:
                if (pick_count == 1):
                    pick_count += 1
                    val = line
                print(val)     
                count += 1
            else:
                print(line)
                
    
    

def main():
    parser = argparse.ArgumentParser(description='Replace test',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input_file', action="store", dest="input_file",default="",help='Input file with codebook indices')
    parser.add_argument('-zero_file', action="store", dest="zero_file",default="",help='Zero file with codebook indices')
    results = parser.parse_args()
    recons_image_from_cb_indices(results) 

if __name__ == '__main__':
    main()
        

