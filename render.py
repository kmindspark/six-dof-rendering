import bpy
import os
import sys
sys.path.append(os.getcwd())
from scipy.spatial.transform import Rotation as R
import json
import time
import bpy, bpy_extras
from math import *
from mathutils import *
import random
import numpy as np
from random import sample
from dr_utils import color_randomize, randomize_light, pattern
import bmesh

'''Usage: blender -b -P render.py'''

def clear_scene():
    '''Clear existing objects in scene'''
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def add_camera_light():
    bpy.ops.object.light_add(type='SUN', radius=1, location=(0,0,0))
    bpy.ops.object.camera_add(location=(0,0,0.8), rotation=(0,0,0))
    bpy.context.scene.camera = bpy.context.object
    return bpy.context.object

def set_render_settings(engine, render_size, generate_masks=True):
    # Set rendering engine, dimensions, colorspace, images settings
    if os.path.exists("./images"):
        os.system('rm -r ./images')
    os.makedirs('./images')
    if os.path.exists("./annots"):
        os.system('rm -r ./annots')
    os.makedirs('./annots')
    scene = bpy.context.scene
    scene.render.resolution_percentage = 100
    scene.render.engine = engine
    render_width, render_height = render_size
    scene.render.resolution_x = render_width
    scene.render.resolution_y = render_height
    scene.use_nodes = True
    scene.render.image_settings.file_format='JPEG'
    scene.view_settings.exposure = 1.3
    if engine == 'BLENDER_WORKBENCH':
        scene.render.image_settings.color_mode = 'RGB'
        scene.display_settings.display_device = 'None'
        scene.sequencer_colorspace_settings.name = 'XYZ'
    elif engine == "BLENDER_EEVEE":
        scene.view_settings.view_transform = 'Raw'
        scene.eevee.taa_samples = 1
        scene.eevee.taa_render_samples = 1
    elif engine == 'CYCLES':   
        scene.render.image_settings.file_format='JPEG'
        #scene.cycles.samples = 50
        scene.cycles.samples = 10
        scene.view_settings.view_transform = 'Raw'
        scene.cycles.max_bounces = 1
        scene.cycles.min_bounces = 1
        scene.cycles.glossy_bounces = 1
        scene.cycles.transmission_bounces = 1
        scene.cycles.volume_bounces = 1
        scene.cycles.transparent_max_bounces = 1
        scene.cycles.transparent_min_bounces = 1
        scene.view_layers["View Layer"].use_pass_object_index = True
        scene.render.tile_x = 16
        scene.render.tile_y = 16

def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px*scale / 2
    v_0 = resolution_y_in_px*scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  ,  alpha_v, v_0),
        (    0  ,    0,      1 )))
    return K

def compute_world_to_camera_matrix(camera):
    if camera.type != 'CAMERA':
        raise Exception("Object {} is not a camera.".format(camera.name))
    # Get the two components to calculate M
    render = bpy.context.scene.render
    modelview_matrix = camera.matrix_world.inverted()
    projection_matrix = camera.calc_matrix_camera(
        bpy.data.scenes["Scene"].view_layers["View Layer"].depsgraph,
        x = render.resolution_x,
        y = render.resolution_y,
        scale_x = render.pixel_aspect_x,
        scale_y = render.pixel_aspect_y,
    )
    # print(projection_matrix * modelview_matrix)
    # Compute P’ = M * P
    transformation_matrix = projection_matrix @ modelview_matrix
    return transformation_matrix


def render(episode):
    bpy.context.scene.render.filepath = "./images/%05d.jpg"%episode
    bpy.ops.render.render(write_still=True)

def get_center_axes(pixel, rot_euler, trans, render_size, transformation_matrix):
    world_to_cam = transformation_matrix
    rot_mat = R.from_euler('xyz', rot_euler).as_matrix()
    #axes = np.eye(3)
    axes = np.float32([[1,0,0],[0,1,0],[0,0,-1]])*0.3
    #axes = np.float32([[1,0,0],[0,1,0]])*0.3
    axes = rot_mat@axes
    axes += trans
    axes_projected = []
    center_projected = project_3d_point(world_to_cam, Vector(trans), render_size)
    for axis in axes:
        axes_projected.append(project_3d_point(world_to_cam, Vector(axis), render_size))
    axes_projected = np.array(axes_projected)
    center_projected = pixel #tuple(center_projected)
    pixel = (200/60)*pixel
    pixel = tuple(pixel.astype(int))
    return pixel, center_projected, axes_projected

def project_3d_point(transformation_matrix,p,render_size):
    p1 = transformation_matrix @ Vector((p.x, p.y, p.z, 1))
    p2 = Vector(((p1.x/p1.w, p1.y/p1.w)))
    p2 = (np.array(p2) - (-1))/(1 - (-1)) # Normalize -1,1 to 0,1 range
    pixel = [int(p2[0] * render_size[0]), int(render_size[1] - p2[1]*render_size[1])]
    return pixel

def get_angle(source_px, vec1_endpoint, d_source_px, vec2_endpoint):
    y1 = vec1_endpoint[0].ravel()
    y2 = vec2_endpoint[0].ravel()
    v1 = y1 - source_px
    v2 = y2 - d_source_px
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(v1, v2)
    angle = np.arccos(dot_product)
    # if angle > (np.pi/2):
    #     angle = angle - np.pi 
    # elif angle < (-np.pi/2):
    #     angle = angle + np.pi
    print(angle)
    return -1*angle
    
def annotate(obj, episode, render_size, transformation_matrix, distractor=None):
    scene = bpy.context.scene
    trans = np.array(obj.matrix_world.translation)
    camera_coord = bpy_extras.object_utils.world_to_camera_view(scene, bpy.context.scene.camera, obj.matrix_world.translation)
    pixel = [round(camera_coord.x * render_size[0]), round(render_size[1] - camera_coord.y * render_size[1])]
    rot_euler = obj.matrix_world.inverted().to_euler()
    if distractor == None:
        metadata = {"trans": trans, "rot": np.array(rot_euler), "pixel":np.array(pixel), "angle": np.array([0.0])} # "d_rot": np.array(rot_euler)
    else:
        # learn orientation for distractor to determine orthogonality
        d_trans = np.array(distractor.matrix_world.translation)
        d_camera_coord = bpy_extras.object_utils.world_to_camera_view(scene, bpy.context.scene.camera, distractor.matrix_world.translation)
        d_pixel = [round(d_camera_coord.x * render_size[0]), round(render_size[1] - d_camera_coord.y * render_size[1])]
        d_rot_euler = distractor.matrix_world.inverted().to_euler()
        _, center_projected, axes_projected = get_center_axes(np.array(pixel), np.array(rot_euler), trans, render_size, transformation_matrix)
        _, d_center_projected, d_axes_projected = get_center_axes(np.array(d_pixel), np.array(d_rot_euler), d_trans, render_size, transformation_matrix)
        angle = get_angle(center_projected, axes_projected, d_center_projected, d_axes_projected)
        metadata = {"trans": trans, "rot": np.array(rot_euler), "pixel":np.array(pixel), "angle": np.array([angle])} #"d_rot": np.array(d_rot_euler)
    np.save('annots/%05d.npy'%episode,metadata) 

def generate_obj():
    bpy.ops.import_mesh.stl(filepath="cyl.stl")
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.subdivide(number_cuts=20)
    bpy.ops.object.editmode_toggle()
    bpy.ops.object.modifier_add(type='SIMPLE_DEFORM')
    bpy.context.object.modifiers["SimpleDeform"].deform_axis = 'Z'
    bpy.context.object.modifiers["SimpleDeform"].deform_method = 'BEND'
    bpy.context.object.modifiers["SimpleDeform"].angle = 0
    obj = bpy.context.object
    obj.pass_index = 1
    return obj

def generate_state(obj, trans_x_range=(0,0), trans_y_range=(0,0), trans_z_range=(0,0),\
                        rot_x_range=(-np.pi/6, np.pi/6), \
                        rot_y_range=(-np.pi/6, np.pi/6), \
                        rot_z_range=(-np.pi/2, np.pi/2)):
    obj.location += Vector((random.uniform(trans_x_range[0], trans_x_range[1]), \
                            random.uniform(trans_y_range[0], trans_y_range[1]), \
                            random.uniform(trans_z_range[0], trans_z_range[1]))) 
    obj.rotation_euler = (random.uniform(rot_x_range[0], rot_x_range[1]), \
                          random.uniform(rot_y_range[0], rot_y_range[1]), \
                          random.uniform(rot_z_range[0], rot_z_range[1])) 
    # this is the curvature / deform angle you want (self note @VAINAVI)
    obj.modifiers["SimpleDeform"].angle = random.uniform(0, 1.5*np.pi)*random.choice((-1,1))
    return obj.location, obj.rotation_euler, obj.modifiers["SimpleDeform"].angle


def generate_table():
    print("here")
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0,0,-0.9))
    table = bpy.context.object 
    #table = colo(table, 'table_texture.png')
    return table

def generate_dataset(iters=1):
    #render_size = (640,480)
    render_size = (60,60)
    #set_render_settings('BLENDER_WORKBENCH', render_size)
    #set_render_settings('BLENDER_EEVEE', render_size)
    set_render_settings('CYCLES', render_size)
    clear_scene()
    table = generate_table()
    camera = add_camera_light()
    transformation_matrix = compute_world_to_camera_matrix(camera)
    num_annotations = 100

    color_white=(194/255., 195/255., 127/255.)
    color_dark = (30/255., 30/255., 30/255.)
    color_red=(120/255., 50/255., 29/255.)

    color_choices = [color_white, color_red]
    #color_choices = [color_white]

    obj = generate_obj()
    distractor_cyl_1 = generate_obj()
    distractor_cyl_1.location = (0, 0, -0.3)
    for episode in range(iters):
        #generate_state(obj, trans_x_range=(-0.03, 0.03), trans_y_range=(-0.03, 0.03))
        generate_state(obj, trans_x_range=(-0.06, 0.06), trans_y_range=(-0.06, 0.06))
        randomize_light()
        color = random.choice(color_choices)
        color_randomize(obj, color)
        color_randomize(table, color_dark)
        color = random.choice(color_choices)
        color_randomize(distractor_cyl_1, color)
        trans_x_range = np.array([0.0,0.05])*random.choice((-1,1))
        trans_y_range = np.array([0.0,0.05])*random.choice((-1,1))
        trans_z_range = (-0.05, -0.075)
        generate_state(distractor_cyl_1, trans_x_range, trans_y_range)
        if random.random() < 0.3:
            distractor_cyl_1.hide_set(True)
            distractor_cyl_1.hide_render = True
            render(episode)
            annotate(obj, episode, render_size, transformation_matrix, distractor=None)
        else:
            distractor_cyl_1.hide_set(False)
            distractor_cyl_1.hide_render = False
            render(episode)
            annotate(obj, episode, render_size, transformation_matrix, distractor = distractor_cyl_1)
        obj.location = np.zeros(3)
    np.save('annots/cam_to_world.npy', np.array(transformation_matrix))

if __name__ == '__main__':
    generate_dataset(750)
