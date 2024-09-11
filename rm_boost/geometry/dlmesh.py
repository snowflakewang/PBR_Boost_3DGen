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

        rm = buffers['rm'][...,0:3] # [0,1]
        mask = buffers['rm'][...,3:4].detach().clone()
        mask = torch.where(mask >= 0.5, 1.0, 0.0)

        rm_refer = target['ks_refer']

        rm_mse_loss = (((rm - rm_refer) ** 2) * mask).sum() / (mask.sum() + 1e-8)
        img_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        return rm_mse_loss, img_loss, reg_loss
    