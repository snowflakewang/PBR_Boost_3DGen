import os
import numpy as np
import open3d as o3d
import imageio.v2 as imageio
import copy
import xatlas
import trimesh
import tqdm
from scipy.spatial.transform import Rotation as R
import pdb

def convert_external_obj_to_std_obj(src_path, dst_path):
    with open(dst_path, 'w') as w:
        attr_dict = {'v': [], 'vt': [], 'vn': [], 'f': []}
        with open(src_path, 'r') as r:
            lines = r.readlines()
            for line in lines:
                if line.split(' ')[0] == 'v':
                    attr_dict['v'].append(line)
                elif line.split(' ')[0] == 'vt':
                    attr_dict['vt'].append(line)
                elif line.split(' ')[0] == 'vn':
                    attr_dict['vn'].append(line)
                elif line.split(' ')[0] == 'f':
                    attr_dict['f'].append(line)
        
        w.write('mtllib mesh.mtl\n')
        w.write('g default\n')
        
        for k in ['v', 'vt', 'vn']:
            for l in attr_dict[k]:
                w.write(l)
        w.write('s 1\n')
        w.write('g pMesh1\n')
        w.write('usemtl defaultMat\n')
        for l in attr_dict['f']:
            w.write(l)

def convert_external_mtl_to_std_mtl(dst_path):
    with open(dst_path, 'w') as w:
        w.write('newmtl defaultMat\n')
        w.write('bsdf   pbr\n')
        w.write('map_Kd texture_kd.png\n')
        w.write('map_Ks texture_ks.png\n')

def resize_crm_obj(src_path, dst_path, zoom_scale=1.3):
    mesh = o3d.io.read_triangle_mesh(src_path)
    nrm_mesh = mesh.compute_vertex_normals()
    nrm_mesh = copy.deepcopy(nrm_mesh)

    x_y_z_max = np.max(np.asarray(nrm_mesh.vertices), axis=0)
    x_y_z_min = np.min(np.asarray(nrm_mesh.vertices), axis=0)
    x_y_z_max_min_distance = x_y_z_max - x_y_z_min
    regular_scale = 1.0 / np.max(x_y_z_max_min_distance) # normalization
    resize_mesh = copy.deepcopy(nrm_mesh)
    resize_mesh.scale(regular_scale * zoom_scale, center=(0, 0, 0)) # center=new_mesh.get_center()

    o3d.io.write_triangle_mesh(dst_path, resize_mesh)

def resize_wonder3d_obj(src_path, dst_path, zoom_scale):
    mesh = o3d.io.read_triangle_mesh(src_path)
    nrm_mesh = copy.deepcopy(mesh)

    x_y_z_max = np.max(np.asarray(nrm_mesh.vertices), axis=0)
    x_y_z_min = np.min(np.asarray(nrm_mesh.vertices), axis=0)
    x_y_z_max_min_distance = x_y_z_max - x_y_z_min
    regular_scale = 1.0 / np.max(x_y_z_max_min_distance) # normalization
    resize_mesh = copy.deepcopy(nrm_mesh)
    resize_mesh.scale(regular_scale * zoom_scale, center=(0, 0, 0)) # center=new_mesh.get_center()

    resize_mesh.create_coordinate_frame()
    rotate_mesh = copy.deepcopy(resize_mesh)

    R = rotate_mesh.get_rotation_matrix_from_xyz((np.pi/2, 0, 0)) #(np.pi/2, 0, 0)
    rotate_mesh.rotate(R, center=(0, 0, 0))

    o3d.io.write_triangle_mesh(dst_path, rotate_mesh)

def resize_instantmesh_obj(src_path, dst_path, zoom_scale=1.0):
    mesh = o3d.io.read_triangle_mesh(src_path)
    nrm_mesh = mesh.compute_vertex_normals()
    nrm_mesh = copy.deepcopy(nrm_mesh)
    
    x_y_z_max = np.max(np.asarray(nrm_mesh.vertices), axis=0)
    x_y_z_min = np.min(np.asarray(nrm_mesh.vertices), axis=0)
    x_y_z_max_min_distance = x_y_z_max - x_y_z_min
    regular_scale = 1.0 / np.max(x_y_z_max_min_distance) # normalization
    resize_mesh = copy.deepcopy(nrm_mesh)
    resize_mesh.scale(regular_scale * zoom_scale, center=(0, 0, 0)) # center=new_mesh.get_center()
    resize_mesh.create_coordinate_frame()
    rotate_mesh = copy.deepcopy(resize_mesh)

    R = rotate_mesh.get_rotation_matrix_from_xyz((-np.pi/2, 0, np.pi/2))
    rotate_mesh.rotate(R, center=(0, 0, 0))

    o3d.io.write_triangle_mesh(dst_path, rotate_mesh)

def resize_triposr_obj(src_path, dst_path, zoom_scale):
    mesh = o3d.io.read_triangle_mesh(src_path)
    nrm_mesh = copy.deepcopy(mesh)

    x_y_z_max = np.max(np.asarray(nrm_mesh.vertices), axis=0)
    x_y_z_min = np.min(np.asarray(nrm_mesh.vertices), axis=0)
    x_y_z_max_min_distance = x_y_z_max - x_y_z_min
    regular_scale = 1.0 / np.max(x_y_z_max_min_distance) # normalization
    resize_mesh = copy.deepcopy(nrm_mesh)
    resize_mesh.scale(regular_scale * zoom_scale, center=(0, 0, 0)) # center=new_mesh.get_center()

    resize_mesh.create_coordinate_frame()
    rotate_mesh = copy.deepcopy(resize_mesh)

    R = rotate_mesh.get_rotation_matrix_from_xyz((0, np.pi/2, 0))
    rotate_mesh.rotate(R, center=(0, 0, 0))

    o3d.io.write_triangle_mesh(dst_path, rotate_mesh)

def resize_sketchfab_obj(src_path, dst_path, zoom_scale, translation_vector, rotate_angle):
    mesh = trimesh.load(src_path, process=False)
    
    # rotate mesh
    angles = rotate_angle
    rotation_x = R.from_euler('x', np.radians(angles['x'])).as_matrix()
    rotation_y = R.from_euler('y', np.radians(angles['y'])).as_matrix()
    rotation_z = R.from_euler('z', np.radians(angles['z'])).as_matrix()
    combined_rotation = rotation_z @ rotation_y @ rotation_x
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = combined_rotation
    mesh.apply_transform(transform_matrix)

    vertices = mesh.vertices
    texture_coords = mesh.visual.uv
    normals = mesh.vertex_normals

    x_y_z_max = np.max(np.asarray(vertices), axis=0)
    x_y_z_min = np.min(np.asarray(vertices), axis=0)
    x_y_z_max_min_distance = x_y_z_max - x_y_z_min
    regular_scale = 1.0 / np.max(x_y_z_max_min_distance) # normalization
    scale_factor = zoom_scale * regular_scale

    # translate mesh
    x_y_z_mid = (x_y_z_max + x_y_z_min) / 2.0
    vertices = vertices - x_y_z_mid

    # scale and translate vertices
    scaled_vertices = vertices * scale_factor
    scaled_vertices = scaled_vertices + translation_vector

    scaled_mesh = trimesh.Trimesh(vertices=scaled_vertices,
                                faces=mesh.faces,
                                vertex_normals=normals,
                                visual=mesh.visual)

    scaled_mesh.export(dst_path)

def mesh_preprocess(src_path, dst_path, model_name):
    if model_name == 'crm':
        resize_crm_obj(src_path=src_path, dst_path=f'{dst_path}/mesh_resize.obj', zoom_scale=2.2)
    elif model_name == 'instantmesh':
        resize_instantmesh_obj(src_path=src_path, dst_path=f'{dst_path}/mesh_resize.obj', zoom_scale=2.2)
    elif model_name == 'wonder3d':
        resize_wonder3d_obj(src_path=src_path, dst_path=f'{dst_path}/mesh_resize.obj', zoom_scale=2.2)
    elif model_name == 'triposr':
        resize_triposr_obj(src_path=src_path, dst_path=f'{dst_path}/mesh_resize.obj', zoom_scale=2.2)
    elif model_name == 'sketchfab':
        # 3D meshes from the Objaverse-XL perhaps need to be triangulated by Blender at first
        resize_sketchfab_obj(
            src_path=src_path, 
            dst_path=f'{dst_path}/mesh_resize.obj',
            zoom_scale=2.5, translation_vector=np.array([0.0, -0.1, 0.0]), rotate_angle={'x': 0, 'y':-0, 'z': 0}
        )
    else:
        raise NotImplementedError
    
    convert_external_obj_to_std_obj(
        src_path=f'{dst_path}/mesh_resize.obj',
        dst_path=f'{dst_path}/mesh_reorg.obj',
    )
    
    convert_external_mtl_to_std_mtl(
        dst_path=f'{dst_path}/mesh.mtl',
    )


if __name__ == '__main__':
    src_path = 'path/to/load/mesh.obj'
    dst_path = 'path/to/save'
    model_name = 'crm' # 'instantmesh', 'wonder3d', 'triposr', 'sketchfab'

    mesh_preprocess(
        src_path=src_path,
        dst_path=dst_path,
        model_name=model_name,
    )