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


def reconstruct_with_dalle(x, encoder, decoder, do_preprocess=False):

  z = torch.LongTensor([1]*48*48)
  z = z.view(1,48,48)
  
  print(f"DALL-E: latent shape: {z.shape}")
  z = F.one_hot(z, num_classes=encoder.vocab_size).permute(0, 3, 1, 2).float()

  x_stats = decoder(z).float()
  x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
  x_rec = T.ToPILImage(mode='RGB')(x_rec[0])

  return x_rec


def stack_reconstructions(input, x0,titles=[]):
  font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf", 22)
  assert input.size == x0.size
  w, h = input.size[0], input.size[1]
  img = Image.new("RGB", (2*w, h))
  img.paste(input, (0,0))
  img.paste(x0, (1*w,0))
  for i, title in enumerate(titles):
    ImageDraw.Draw(img).text((i*w, 0), f'{title}', (255, 255, 255), font=font) # coordinates, text, color, font
  return img

def reconstruction_pipeline(encoder_dalle, decoder_dalle,url,device, size=320,is_local=False):
    titles=["Input", "DALL-E dVAE (f8, 8192)"]

    x5 = reconstruct_with_dalle(x_dalle, encoder_dalle, decoder_dalle)
    img = stack_reconstructions(custom_to_pil(preprocess_vqgan(x_vqgan[0])), x5, 
                              titles=titles)

def output_dalle_codebook_viz(params):
    sys.path.append(".")

    # also disable grad to save memory
    torch.set_grad_enabled(False)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #DEVICE = torch.device("cpu")



    encoder = load_model("./dalle_models/encoder.pkl", DEVICE)
    decoder = load_model("./dalle_models/decoder.pkl", DEVICE)
    print(f"Encode vocab size: {encoder.vocab_size}")

    dim = 48
    total = 8192
    for i in range(total): 
        output_file = params.output + "/" + str(i+1) + ".png"
        try:
            fp = open(output_file)
            print("Already generated",output_file)
            fp.close()
            continue
        except:
            pass
        z = torch.LongTensor([i]*dim*dim)
        z = z.to(DEVICE)
        z = z.view(1,dim,dim)
  
        z = F.one_hot(z, num_classes=encoder.vocab_size).permute(0, 3, 1, 2).float()

        x_stats = decoder(z).float()
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        x_rec = T.ToPILImage(mode='RGB')(x_rec[0])
        output_file = params.output + "/" + str(i+1) + ".png"
        print(output_file)
        x_rec.save(output_file)


   




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dalles codebook visualization',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-output', action="store", dest="output",default="dalle_codebook",help='Output directory of dalle codevbook visualizations')
    results = parser.parse_args()
    output_dalle_codebook_viz(results)

