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

from dall_e          import map_pixels, unmap_pixels, load_model
from IPython.display import display, display_markdown


def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

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

def reconstruct_with_vqgan(x, model):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  z, _, [_, _, indices] = model.encode(x)
  print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
  xrec = model.decode(z)
  return x,z,xrec
  #return x,indices,xrec #Note cosine distance in codebook space clusters all near 1



def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return PIL.Image.open(io.BytesIO(resp.content))


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




#def stack_reconstructions(input, x0, x1, x2, x3,x4,x5, titles=[]):
def stack_reconstructions(input, x1, titles=[]):
  font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf", 22)
  assert input.size == x1.size
  w, h = input.size[0], input.size[1]
  img = Image.new("RGB", (2*w, h))
  img.paste(input, (0,0))
  img.paste(x1, (1*w,0))
  for i, title in enumerate(titles):
    ImageDraw.Draw(img).text((i*w, 0), f'{title}', (255, 255, 255), font=font) # coordinates, text, color, font
  return img


#This was abandoned
def custom_flatten(z):
    z = rearrange(z, 'b c h w -> b h w c').contiguous()
    z_flattened = z.view(-1,3)
    d = torch.sum(z_flattened, dim=1, keepdim=True)
    return d.flatten().squeeze().tolist()

#def reconstruction_pipeline(modelvqf4, modelvqf4noattn, modelvqf8,modelvqf8n256,modelvqf16,encoder_dalle, decoder_dalle,url,device, size=320,is_local=False):
def reconstruction_pipeline(modelvqf4noattn,url,device, size=320,is_local=False):
    titles=["Input", "Reconstructed(VQF4)"] #no attn

    if (is_local):
        x_vqgan = preprocess(PIL.Image.open(url), target_image_size=size, map_dalle=False)
    else:
        x_vqgan = preprocess(download_image(url), target_image_size=size, map_dalle=False)
    x_vqgan = x_vqgan.to(device)
    print(f"input is of size: {x_vqgan.shape}")
    inp,encoded,x1 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), modelvqf4noattn)
    img = stack_reconstructions(custom_to_pil(preprocess_vqgan(x_vqgan[0])),
                              custom_to_pil(x1[0]), 
                              titles=titles)
    return img,inp.flatten().tolist(),encoded.flatten().tolist(),x1.flatten().tolist()
    #return img,custom_flatten(inp),custom_flatten(encoded),custom_flatten(x1)

def load_model(params):
    sys.path.append(".")

    # also disable grad to save memory
    torch.set_grad_enabled(False)

    #DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = torch.device("cpu")



    configvqf4noattn = my.load_config("models/first_stage_models/vq-f4-noattn/config.yaml", display=False)
    model = my.load_vqgan(configvqf4noattn, ckpt_path="models/first_stage_models/vq-f4-noattn/model.ckpt", is_gumbel=False).to(DEVICE)

    return model

def reconstruct(model,input_file,output_file,size):
    DEVICE = torch.device("cpu")
    # also disable grad to save memory
    torch.set_grad_enabled(False)

    img,inp,encoded,recons = reconstruction_pipeline(model, url=input_file,device=DEVICE, size=size,is_local=True)
    img.save(output_file)
    return inp,encoded,recons

def make_dir(path):
    try:
        os.mkdir(path)
    except:
        print("Output directory:", path," already exists") 


def save_vectors(vecs,out_file):
    with open(out_file,"wb") as fp:
        np.save(fp,np.array(vecs))

def file_already_exists(output_file):
    try:
        fp = open(output_file)
        fp.close()
        return True
    except:
        return False
    

def batch_gen(params):
    sys.path.append(".")
    input_dir = params.input
    output_dir = params.output
    model = load_model(params)
    make_dir(output_dir)
    recons_images_dir = output_dir + "/recons_images"
    make_dir(recons_images_dir)
    log_info_file = output_dir +  "/index.txt"
    log_fp = open(log_info_file,"w")
    inp_vecs = []
    encoded_vecs = []
    recons_vecs = []
    count = 1
    for file_names in os.listdir(input_dir):
        input_file = os.path.join(input_dir, file_names)
        if  os.path.isfile(input_file):
            if (input_file.endswith(".jpeg") or input_file.endswith(".jpg") or input_file.endswith(".png")):
                print(f"{count} ] {input_file}")
                count += 1
                file_only = input_file.split("/")[-1]
                output_file =  recons_images_dir + "/" +  file_only
                if (file_already_exists(output_file)):
                    print(f"File {output_file} already exists. Skipping Generation")
                    continue;
                inp,encoded,recons = reconstruct(model,input_file,output_file,params.size)
                inp_vecs.append(inp)
                encoded_vecs.append(encoded)
                recons_vecs.append(recons)
                out_str = f"{file_only}\n"
                log_fp.write(out_str)
    log_fp.close()
    assert(len(inp_vecs) == len(encoded_vecs) == len(recons_vecs))
    save_vectors(inp_vecs,output_dir + "/input_vectors.npy")
    save_vectors(encoded_vecs,output_dir + "/encoded_vectors.npy")
    save_vectors(recons_vecs,output_dir + "/recons_vectors.npy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batched Reconstruction and output of vectors',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input', action="store", dest="input",required=True,help='Input directory')
    parser.add_argument('-output', action="store", dest="output",required=True,help='Output directory')
    parser.add_argument('-size', action="store", dest="size",default=384,type=int,help='Expected size. Do not change this default for VQ models')
    results = parser.parse_args()
    batch_gen(results)

