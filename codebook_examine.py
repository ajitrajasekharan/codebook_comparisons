import sys
import yaml
import torch
from omegaconf import OmegaConf
import pdb
from taming.models.vqgan import VQModel, GumbelVQ
import io
import os, sys
import requests
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np
import argparse
from einops import rearrange
import my_utils as my

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from IPython.display import display, display_markdown

def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1,2,0).numpy()
  x = (255*x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

def load_model(params):
    sys.path.append(".")

    # also disable grad to save memory
    torch.set_grad_enabled(False)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #DEVICE = torch.device("cpu")



    configvqf4noattn = my.load_config("models/first_stage_models/vq-f4-noattn/config.yaml", display=False)
    model = my.load_vqgan(configvqf4noattn, ckpt_path="models/first_stage_models/vq-f4-noattn/model.ckpt", is_gumbel=False).to(DEVICE)

    return model


def save_vectors(vecs,out_file):
    with open(out_file,"wb") as fp:
        np.save(fp,np.array(vecs))

def codebook_examine (params):
    output_file = params.output
    model = load_model(params)
    log_fp = open(output_file,"w")
    vecs = model.quantize.embedding.weight.tolist()
    save_vectors(vecs,output_file)
    with open("index.txt","w") as fp:
        for i in range(len(vecs)):
            fp.write(str(i+1) + "\n")

def recons_using_codebook_values(params):
    output_dir = params.output
    try:
        os.mkdir(output_dir)
    except:
        print("Output directory:", output_dir," already exists") 
    model = load_model(params)
    vecs = model.quantize.embedding.weight.tolist()
    codebook_size = len(vecs)
    for i in range(codebook_size):
        z = model.quantize.get_codebook_entry(torch.LongTensor([i]*96*96),None)
        z = z.permute(1,0).contiguous().view(1,3,96,96)
        xrec = model.decode(z)
        xrec = custom_to_pil(xrec[0])
        w, h = xrec.size[0],xrec.size[0]
        img = Image.new("RGB", (w, h))
        img.paste(xrec, (0,0))
        output_file = f"{output_dir}/{i+1}.png"
        img.save(output_file)

def preprocess(img, target_image_size=256, map_dalle=True):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    if map_dalle: 
      img = map_pixels(img)
    return img

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

def gen_image_using_image_codebook_values(input_file,recons_output_file,model):
    arr = []
    with open(input_file) as fp:
        for line in fp:
            line = int(line.rstrip("\n"))
            arr.append(line)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pdb.set_trace()
    arr = torch.LongTensor(arr)
    arr = arr.to(device)
    z = model.quantize.get_codebook_entry(arr,None)
    z = z.permute(1,0).contiguous().view(1,3,96,96)
    xrec = model.decode(z)
    xrec = custom_to_pil(xrec[0])
    w, h = xrec.size[0],xrec.size[0]
    img = Image.new("RGB", (w, h))
    img.paste(xrec, (0,0))
    img.save(recons_output_file)

def gen_image_using_image_custom_codes(params):
    input_file = params.input
    output_file = "regen_" +  ''.join(input_file.split(".")[:-1]) +  ".png"
    model = load_model(params)
    arr = []
    index_correct = 1
    with open(input_file) as fp:
        for line in fp:
            line = int(line.rstrip("\n")) - index_correct 
            arr.append(line)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arr = torch.LongTensor(arr)
    arr = arr.to(device)
    z = model.quantize.get_codebook_entry(arr,None)
    z = z.permute(1,0).contiguous().view(1,3,96,96)
    xrec = model.decode(z)
    xrec = custom_to_pil(xrec[0])
    w, h = xrec.size[0],xrec.size[0]
    img = Image.new("RGB", (w, h))
    img.paste(xrec, (0,0))
    img.save(output_file)
            

def output_image_codes(params):
    input_file = params.input
    recons_output_file =  "codbook_recons_" + input_file.split("/")[-1]
    output_file =  "indices_" + input_file.split("/")[-1]
    output_file = ''.join(output_file.split(".")[:-1]) +  ".txt"
    model = load_model(params)
    pdb.set_trace()
    x_vqgan = preprocess(PIL.Image.open(input_file).convert("RGB"), target_image_size=params.size, map_dalle=False)
   # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_vqgan = x_vqgan.to(device)
    x = preprocess_vqgan(x_vqgan)
    z, _, [_, _, indices] = model.encode(x)
    with open(output_file,"w") as fp:
        for i in range(len(indices)):
            fp.write(str(indices[i].tolist()) + "\n")
    gen_image_using_image_codebook_values(output_file,recons_output_file,model)
        

def batched_output_image_codes(params):
    input_dir = params.input
    output_dir = params.output
    model = load_model(params)
    #device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    count = 1
    for file_names in os.listdir(input_dir):
        input_file = os.path.join(input_dir, file_names)
        if  os.path.isfile(input_file):
            if (input_file.endswith(".jpeg") or input_file.endswith(".jpg") or input_file.endswith(".png")):
                print(f"{count} ] {input_file}")
                count += 1
                file_only = input_file.split("/")[-1]
                file_only = ''.join(file_only.split(".")[:-1])
                output_file =  output_dir + "/" +  file_only
                x_vqgan = preprocess(PIL.Image.open(input_file).convert("RGB"), target_image_size=params.size, map_dalle=False)
                x_vqgan = x_vqgan.to(device)
                x = preprocess_vqgan(x_vqgan)
                z, _, [_, _, indices] = model.encode(x)
                with open(output_file,"w") as fp:
                    for i in range(len(indices)):
                        fp.write(str(indices[i].tolist()) + "\n")


    
    

def main():
    parser = argparse.ArgumentParser(description='Codebook examine',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input', action="store", dest="input",default="",help='Input file for some options')
    parser.add_argument('-output', action="store", dest="output",default="codebook",help='Output file/dir for codebook value images')
    parser.add_argument('-size', action="store", dest="size",default=384,type=int,help='Expected size. Do not change this default for VQ models')
    results = parser.parse_args()
    options = "Enter option:\n\t(1) Examine codebook - dump codebook vectors\n\t(2) Visualize each codebook values.Write images into out dir\n\t(3) generate image using custom codes\n\t(4) Output codebook indices for an input image\n\t(5) Output codebook indices for all images in input dir\n\t(0) Quit"
    #output_image_codes(results)
    #recons_using_codebook_values(results)
    #gen_image_using_image_custom_codes(results)
    while (True):
        print(options)
        inp = int(input())
        if (inp == 1):
            codebook_examine(results)
            break
        elif (inp == 2):
            recons_using_codebook_values(results)
        elif (inp == 3):
            gen_image_using_image_custom_codes(results)
        elif (inp == 4):
            output_image_codes(results)
        elif (inp == 5):
            batched_output_image_codes(results)
        else:
            if (inp == 0):
                print("Quitting")
                break
            print(options) 

if __name__ == '__main__':
    main()
        

