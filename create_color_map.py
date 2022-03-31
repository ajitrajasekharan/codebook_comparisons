import os
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np
import argparse
import torchvision.transforms as transforms
import pdb

def rescale(input_file,width):
    trans = transforms.Compose([
        transforms.Resize(width),
        transforms.CenterCrop(width)
      ])
    transformed_sample = trans(Image.open(input_file)).convert('RGB')
    return transformed_sample

def tile_images(params):
    input_dir = params.input
    output_file = params.output
    width  = params.size
    height  = width
    x_pos = 0
    y_pos = 0
    max_slot = 96
    image_height = (int((params.count/max_slot) + 1)* width )
    img = Image.new("RGB", ((max_slot)*width,image_height))
    for i in range(params.count):
        scaled_image = rescale(f"{input_dir}/{i+1}.png",width) 
        img.paste(scaled_image, (x_pos,y_pos))
        x_pos += width
        if ((x_pos/width) % max_slot == 0):
            print(f"new y_pos :{y_pos}")
            x_pos = 0
            y_pos += width
    img.save(output_file)

    
def read_indices(input_file):
    arr = []
    with open(input_file) as fp:
        for line in fp:
            line = int(line.rstrip("\n"))
            arr.append(line)
    return arr
    


def recons_image_from_cb_indices_impl(input_dir,input_file,output_file,size):
    width  = size
    height  = width
    x_pos = 0
    y_pos = 0
    max_slot = 96
    image_height = width
    img = Image.new("RGB", ((max_slot)*width,(max_slot)*width))
    arr = read_indices(input_file)
    for i in range(len(arr)):
        index = arr[i]
        scaled_image = rescale(f"{input_dir}/{index}.png",width) 
        img.paste(scaled_image, (x_pos,y_pos))
        x_pos += width
        if ((x_pos/width) % max_slot == 0):
            print(f"new y_pos :{y_pos}")
            x_pos = 0
            y_pos += width
    img.save(output_file)

def recons_image_from_cb_indices(params):
    recons_image_from_cb_indices_impl(params.input,params.input_file,params.output,params.size)

    
    
def batch_recons_image_from_cb_indices(params):
    pdb.set_trace()
    codebook_dir = params.input #directory of codebook images
    input_dir = params.input_file #directory of codebook images
    output_dir = params.output
    count = 1
    for file_names in os.listdir(input_dir):
        input_file = os.path.join(input_dir, file_names)
        if  os.path.isfile(input_file):
            print(f"{count} ] {input_file}")
            count += 1
            file_only = input_file.split("/")[-1] + ".png"
            #file_only = ''.join(file_only.split(".")[:-1]) + ".png"
            output_file =  output_dir + "/" +  file_only 
            recons_image_from_cb_indices_impl(codebook_dir,input_file,output_file,params.size)


def main():
    parser = argparse.ArgumentParser(description='Tile image',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input', action="store", dest="input",default="codebook",help='Input dir')
    parser.add_argument('-input_file', action="store", dest="input_file",default="",help='Input file with codebook indices. This could be a dir in batch mode')
    parser.add_argument('-output', action="store", dest="output",default="codebook_color_view2.png",help='Tiled image constrcuted from codebook values. Could be dir in batchmode ')
    parser.add_argument('-size', action="store", dest="size",default=60,type=int,help='Scaled down width of each image')
    parser.add_argument('-count', action="store", dest="count",default=8192,type=int,help='Number of images in dir. This is used to create image file names')
    results = parser.parse_args()
    while True:
        print("Enter option: \n\t1. Create a tile of images from a codebook directory containing images for each code\n\t2. Reconstruct  a single image from CB indices.\n\t3. Reconstruct multiple images in a directory deom CB indicers\n\tq/Q - Quit")
        option = input()
        if (option == "1"):
            tile_images(results)
        elif (option == "2"):
            recons_image_from_cb_indices(results) 
        elif (option == "3"):
            batch_recons_image_from_cb_indices(results) 
        elif (option.lower().startswith("q")):
            break
        else:
            print("Invalid option")

if __name__ == '__main__':
    main()
        

