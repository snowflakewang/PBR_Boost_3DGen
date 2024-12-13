# Boosting 3D Object Generation through PBR Materials

Official implementation for _Boosting 3D Object Generation through PBR Materials_.

## [Project Page](https://snowflakewang.github.io/PBR_Boost_3DGen/) | [arXiv](https://arxiv.org/abs/2411.16080) | [Paper](https://dl.acm.org/doi/10.1145/3680528.3687676) | [Weights](https://huggingface.co/SnowflakeWang/MonoIntrinsics)

## Install

### Step 1 - Base
System requirements: Ubuntu 18.04

Tested GPUs: NVIDIA A100
```bash
conda create -n pbrboost python=3.9
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Step 2 - Image-to-3D (Optional)
We recommend you to install Image-to-3D methods according to their official repositories. We have tested [CRM](https://github.com/thu-ml/CRM), [InstantMesh](https://github.com/TencentARC/InstantMesh), [Wonder3D](https://github.com/xxlong0/Wonder3D), [Era3D](https://github.com/pengHTYX/Era3D), and [TripoSR](https://github.com/VAST-AI-Research/TripoSR).

If you already have a mesh with an Albedo (RGB) UV and just want to apply the boosting method to it, you can skip this step.

## Prepare folder structure

### Step 1 - Structure overview
```bash
3DGen_PBR_Boost
|-- albedo_mesh_gen
    |-- CRM
    |-- MonoAlbedo
    |-- MonoNormal

|-- ckpts
    |-- MonoAlbedo
    |-- MonoNormal

|-- data
    |-- irrmaps
        |-- bsdf_256_256.bin
        |-- modern_buildings_2_4k.hdr # you can add more environmental maps here
    |-- textures
        |-- texture_ks.png
        |-- texture_n.png

|-- normal_boost
    |-- normal_boost.py
    |-- configs
        |-- normal_boost_cfg.json
    |-- input
        |-- XXX # folder name
            |-- dmtet_mesh
                |-- mesh_reorg.obj # converted from the original mesh
                |-- mesh.mtl
                |-- mesh.obj # original mesh, not necessary
                |-- texture_kd.png # albedo UV
                |-- texture_ks.png # copied from data/textures/texture_ks.png
    |-- out
        |-- XXX_out
            |-- dmtet_mesh
                |-- texture_n.png # generated bump UV

|-- rm_boost
    |-- rm_boost.py
    |-- configs
        |-- rm_boost_cfg.json
    |-- input
        |-- XXX # folder name
            |-- dmtet_mesh
                |-- mesh_reorg.obj # converted from the original mesh
                |-- mesh.mtl
                |-- mesh.obj # original mesh, not necessary
                |-- texture_kd.png # albedo UV
                |-- texture_ks.png # copied from data/textures/texture_ks.png
                |-- texure_n.png # copied from data/textures/texture_n.png or normal_boost's out
            |-- ks_mask
                |-- ks
                    |-- val_000000_ks.png
                    |-- val_000025_ks.png
                    |-- val_000050_ks.png
                    |-- val_000075_ks.png
                    |-- val_000100_ks.png
                    |-- val_000101_ks.png
    |-- out
        |-- XXX_out
            |-- dmtet_mesh
                |-- texture_ks.png # generated roughness & metalness UV, [R,G,B] <-> [Zero,Roughness,Metalness]

|-- relight
    |-- relight.py
    |-- configs
        |-- relight_cfg.json
    |-- input
        |-- XXX # folder name
            |-- dmtet_mesh
                |-- mesh_reorg.obj # converted from the original mesh
                |-- mesh.mtl
                |-- texture_kd.png # albedo UV
                |-- texture_ks.png # generated R & M UV, copied from the corresponding folder in rm_boost's out
                |-- texure_n.png # generated bump UV, copied from the corresponding folder in normal_boost's out
    |-- out
        |-- XXX_out
            |-- validate
                |-- kd
                |-- ks
                |-- normal
                |-- mask
                |-- shaded

|-- utils
    |-- mesh_convert.py
``` 

### Step 2 - Download checkpoints
```bash
ckpts/MonoAlbedo
ckpts/MonoNormal
```

### Step 3 - Download other necessary files
```bash
data
input examples of normal_boost/rm_boost/relight
```

## Run

**The following boosting methods are independent from each other. You can only use one of them.**

**Notice: _Roughness & Metalness boosting_ is a semi-automatic process. If you want to use this function, it is necessary to prepare the _ks_mask_ folder as illustrated in the _Step 1 - Structure overview_.**

### Feature 1 - Single image inference
```bash
cd albedo_mesh_gen
# single image-to-albedo
python MonoAlbedo/albedo_infer.py # remember to modify the path of input images

# single image-to-normal
python MonoNormal/normal_infer.py # remember to modify the path of input images
```

### Feature 2 - Mesh & albedo UV generation
```bash
# optional, if you already have a mesh with an Albedo (RGB) UV, you can skip this step
# take CRM as an example, InstantMesh/Wonder3D/Era3D/TripoSR also can be used
cd albedo_mesh_gen/CRM
bash run.sh # remember to modify the path of input images
```
After obtaining a mesh with an albedo (RGB) UV, you can first convert its format to the one that can be processed in the following boosting stages.
```bash
cd utils
python mesh_convert.py # remember to modify the path of input meshes
```
After obtaining a mesh with the expected format, you can put it in the _input_ folder as illustrated in the _Step 1 - Structure overview_.

### Feature 3 - Normal boosting
```bash
cd normal_boost
python normal_boost.py --config configs/normal_boost_cfg.json
```

### Feature 4 - Roughness & Metalness boosting
```bash
cd rm_boost
python rm_boost.py --config configs/rm_boost_cfg.json
```

### Feature 5 - Relighting
```bash
cd relight
python relight.py --config configs/relight_cfg.json
```

## Acknowledgments
- [diffusers](https://github.com/huggingface/diffusers)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
- [CRM](https://github.com/thu-ml/CRM)
- [InstantMesh](https://github.com/TencentARC/InstantMesh)
- [Wonder3D](https://github.com/xxlong0/Wonder3D)
- [TripoSR](https://github.com/VAST-AI-Research/TripoSR)
- [Fantasia3D](https://github.com/Gorilla-Lab-SCUT/Fantasia3D)
- [DreamCraft3D](https://github.com/deepseek-ai/DreamCraft3D)
- [Marigold](https://github.com/prs-eth/Marigold)

## Citation

```
@proceedings{
}
```