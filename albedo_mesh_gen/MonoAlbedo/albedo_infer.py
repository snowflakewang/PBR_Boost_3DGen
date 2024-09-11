import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import datasets
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import diffusers
from albedo_pipeline import MaterialPipeline
import imageio.v2 as imageio
import pdb

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def single_inference(src_path, dst_path, checkpoint):
    pipeline = MaterialPipeline.from_pretrained(
            pretrained_model_name_or_path=checkpoint
        ).to(device)

    image = Image.open(src_path)

    res = pipeline(input_image=image, denoising_steps=30, ensemble_size=1)
    albedo_color = res.albedo_pil
    albedo_color.save(dst_path)

if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    ckpts_path = os.path.join(current_dir, "../../ckpts/MonoAlbedo")
    
    single_inference(
        src_path='xxx.png',
        dst_path='yyy.png',
        checkpoint=ckpts_path,
    )