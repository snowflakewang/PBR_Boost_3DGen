# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch

from render import mesh
from render import render
from render import regularizer
from render import util
from torch.cuda.amp import custom_bwd, custom_fwd 
import numpy as np
import imageio.v2 as imageio
import pdb
import copy
import einops

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None
    
###############################################################################
#  Geometry interface
###############################################################################

class DLMesh(torch.nn.Module):
    def __init__(self, initial_guess, FLAGS):
        super(DLMesh, self).__init__()

        self.FLAGS = FLAGS

        self.initial_guess = initial_guess
        self.mesh          = initial_guess.clone()
        print("Base mesh has %d triangles and %d vertices." % (self.mesh.t_pos_idx.shape[0], self.mesh.v_pos.shape[0]))
        
        # self.mesh.v_pos = torch.nn.Parameter(self.mesh.v_pos, requires_grad= True)
        # self.register_parameter('vertex_pos', self.mesh.v_pos)

     

    @torch.no_grad()
    def getAABB(self):
        return mesh.aabb(self.mesh)

    def getMesh(self, material):
        self.mesh.material = material

        imesh = mesh.Mesh(base=self.mesh)
        # Compute normals and tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)
        return imesh

    def render(self, glctx, target, lgt, opt_material, bsdf=None,if_normal=False, mode = 'appearance_modeling', if_flip_the_normal = False, if_use_bump = False):
        opt_mesh = self.getMesh(opt_material)
        return render.render_mesh(glctx, 
                                  opt_mesh,
                                  target['mvp'],
                                  target['campos'],
                                  lgt,
                                  target['resolution'], 
                                  spp=target['spp'], 
                                  msaa=True,
                                  background= target['background'] ,
                                  bsdf= bsdf,
                                  if_normal=if_normal,
                                  normal_rotate=target['normal_rotate'], 
                                  mode = mode,
                                  if_flip_the_normal = if_flip_the_normal,
                                  if_use_bump = if_use_bump
                                   )

    def tick(self, glctx, target, lgt, opt_material, iteration, if_normal, guidance,  mode, if_flip_the_normal, if_use_bump):
        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        buffers= self.render(glctx, target, lgt, opt_material, if_normal = if_normal, mode = mode,  if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump)
        
        if self.FLAGS.guidance_type == 'rgb2normal_mse':
            nrm_world = buffers['normal'][...,0:3] # [-1,1]

            mask = buffers['shaded'][...,3:4].detach().clone()
            mask = torch.where(mask >= 0.5, 1.0, 0.0)

            srgb = buffers['albedo'][...,0:3]
            srgb = util.rgb_to_srgb(srgb)
            
            mv = target['mv'][:, 0:3, 0:3]
            b, h, w, c = nrm_world.shape
            nrm_cam = torch.bmm(mv, nrm_world.permute(0, 3, 1, 2).reshape(b, c, -1)).reshape(b,c,h,w).permute(0, 2, 3, 1)

            srgb = srgb * 2.0 - 1.0 # [0,1]->[-1,1]

            num_train_timesteps = guidance.scheduler.config.num_train_timesteps
            alphas = guidance.scheduler.alphas_cumprod.to(guidance.device)

            t = torch.ones([self.FLAGS.batch], dtype=torch.long, device='cuda') * int(num_train_timesteps * 0.98)
            #t = torch.randint(int(num_train_timesteps * 0.1), int(num_train_timesteps * 0.98)+1, [self.FLAGS.batch], dtype=torch.long, device='cuda') # [B]
            mask = mask.permute(0, 3, 1, 2).contiguous()
            srgb = srgb.permute(0, 3, 1, 2).contiguous() # [1,3,H,W]
            srgb_latent = guidance._encode_rgb(srgb)

            nrm_cam = nrm_cam.permute(0, 3, 1, 2).contiguous() # [1,3,H,W]
            nrm_cam_latent = guidance._encode_rgb(nrm_cam)

            guidance._encode_empty_text()
            batch_empty_text_embed = guidance.empty_text_embed.repeat((srgb_latent.shape[0], 1, 1))  # [B, 2, 1024]

            num_infer_steps = 50 #10
            guidance.scheduler.set_timesteps(num_infer_steps, device='cuda') #num_inference_steps, 10 can be changed
            infer_timesteps = guidance.scheduler.timesteps  # [T]

            init_t = int(num_infer_steps * 0.2) # 0.2 can be changed
        
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(nrm_cam_latent)
                nrm_cam_latent_noisy = guidance.scheduler.add_noise(nrm_cam_latent, noise, infer_timesteps[init_t])
                # pred noise
                for t in infer_timesteps[init_t:]:
                    unet_input = torch.cat([srgb_latent, nrm_cam_latent_noisy], dim=1)
                    noise_pred = guidance.unet(unet_input, t, encoder_hidden_states=batch_empty_text_embed).sample
                    nrm_cam_latent_noisy = guidance.scheduler.step(noise_pred, t, nrm_cam_latent_noisy).prev_sample
                nrm_cam_denoise = guidance._decode_normal(nrm_cam_latent_noisy)
                nrm_cam_denoise = torch.clip(nrm_cam_denoise, -1.0, 1.0)         
            
            nrm_cam_mse_loss = (((nrm_cam - nrm_cam_denoise) ** 2) * mask).sum() / (mask.sum() + 1e-8)
            img_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
            reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        
            return nrm_cam_mse_loss, img_loss, reg_loss
        else:
            raise NotImplementedError('[INFO] Invalid guidance type.')
    