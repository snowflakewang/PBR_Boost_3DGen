import numpy as np
import torch
import random
from PIL import Image
from normal_pipeline import MonoNormPipeline
from diffusers.utils import load_image
import cv2
import os
from glob import glob
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def single_inference(src_path, dst_name, checkpoint):
    pipe = MonoNormPipeline.from_pretrained(pretrained_model_name_or_path=checkpoint).to(device)

    image = load_image(src_path)

    pipeline_output = pipe(
            image,                  # Input image.
            denoising_steps=30,     # (optional) Number of denoising steps of each inference pass. Default: 10.
            ensemble_size=3,       # (optional) Number of inference passes in the ensemble. Default: 10.
            processing_res=512,     # (optional) Maximum resolution of processing. If set to 0: will not resize at all. Defaults to 768.
            match_input_res=True,   # (optional) Resize depth prediction to match input resolution.
            batch_size=0,           # (optional) Inference batch size, no bigger than `num_ensemble`. If set to 0, the script will automatically decide the proper batch size. Defaults to 0.
            color_map="Spectral",   # (optional) Colormap used to colorize the depth map. Defaults to "Spectral".
            show_progress_bar=True, # (optional) If true, will show progress bars of the inference progress.
        )
    
    uncertainty: torch.Tensor = pipeline_output.uncertainty                    # Predicted uncertainty map
    normal: np.ndarray = pipeline_output.normal_np                    # Predicted depth map
    normal_colored: Image.Image = pipeline_output.normal_pil      # Colorized prediction

    # Save colorized depth map
    normal_colored.save(f'{dst_name}_normal.png')

    if uncertainty is not None:
        uncertainty = (uncertainty * 255).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
        uncertainty = cv2.applyColorMap(uncertainty, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(f'{dst_name}_uncerty.png', uncertainty)

if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    ckpts_path = os.path.join(current_dir, "../../ckpts/MonoNormal")

    single_inference(
        src_path='xxx.png',
        dst_name='yyy',
        checkpoint=ckpts_path,
    )    