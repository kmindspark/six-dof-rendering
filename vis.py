from cv2 import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np
import argparse
import os
import math
import json
import colorsys
from math import *
from mathutils import *

def draw(img, source_px, imgpts):
    imgpts = imgpts.astype(int)
    img = cv2.line(img, tuple(source_px), tuple(imgpts[0].ravel()), (255,0,0), 2)
    img = cv2.line(img, tuple(source_px), tuple(imgpts[1].ravel()), (0,255,0), 2)
    img = cv2.line(img, tuple(source_px), tuple(imgpts[2].ravel()), (0,0,255), 2)
    return img

def draw_distractor(img, source_px, imgpts):
    imgpts = imgpts.astype(int)
    img = cv2.line(img, tuple(source_px), tuple(imgpts[0].ravel()), (100,0,0), 2)
    return img

def draw_angle(img, source_px, imgpts, angle):
    imgpts = imgpts.astype(int)
    img = cv2.line(img, tuple(source_px), tuple(imgpts[0].ravel()), (255,0,0), 2)
    img = cv2.line(img, tuple(source_px), tuple(imgpts[1].ravel()), (0,255,0), 2)
    img = cv2.line(img, tuple(source_px), tuple(imgpts[2].ravel()), (0,0,255), 2)
    other_point = rotate_around_point(tuple(imgpts[0].ravel()), angle, tuple(source_px))
    img = cv2.line(img, tuple(source_px), other_point, (100,0,0), 2)
    return img

def rotate_around_point(xy, radians, origin=(0,0)):
    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = x - offset_x
    adjusted_y = y - offset_y
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y - sin_rad * adjusted_x + cos_rad * adjusted_y
    qx = int(qx)
    qy = int(qy)
    return tuple((qx, qy))

def project_3d_point(transformation_matrix,p,render_size):
    p1 = transformation_matrix @ Vector((p.x, p.y, p.z, 1))
    p2 = Vector(((p1.x/p1.w, p1.y/p1.w)))
    p2 = (np.array(p2) - (-1))/(1 - (-1)) # Normalize -1,1 to 0,1 range
    pixel = [int(p2[0] * render_size[0]), int(render_size[1] - p2[1]*render_size[1])]
    return pixel

def get_center_axes(pixel, rot_euler, trans, render_size):
    world_to_cam = Matrix(np.load('annots/cam_to_world.npy'))
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
    center_projected = pixel.astype(int) #tuple(center_projected)
    pixel = (200/60)*pixel
    pixel = tuple(pixel.astype(int))
    return pixel, center_projected, axes_projected

def show_annots(idx, save=True):
    image_filename = "images/%05d.jpg"%idx
    img = cv2.imread(image_filename)
    H,W,C = img.shape
    render_size = (W,H)
    metadata = np.load("annots/%05d.npy"%idx, allow_pickle=True)
    trans = metadata.item().get("trans")
    rot_euler = metadata.item().get("rot")
    pixel = metadata.item().get("pixel")
    angle = metadata.item().get("angle")
    #d_trans = metadata.item().get("d_trans")
    #d_rot_euler = metadata.item().get("d_rot")
    #d_pixel = metadata.item().get("d_pixel")
    pixel, center_projected, axes_projected = get_center_axes(pixel, rot_euler, trans, render_size)
    #_, _, d_axes_projected = get_center_axes(d_pixel, d_rot_euler, d_trans, render_size)
    vis = img.copy()
    #vis = draw(vis,center_projected,axes_projected)
    #vis = draw_distractor(vis,center_projected,d_axes_projected)
    vis = draw_angle(vis, center_projected, axes_projected, angle)
    vis = cv2.resize(vis,(200,200))
    vis = cv2.circle(vis, pixel, 5, (0,0,0), -1)
    print("Annotating %06d"%idx)
    if save:
    	annotated_filename = "%05d.jpg"%idx
    	cv2.imwrite('./vis/{}'.format(annotated_filename), vis)

if __name__ == '__main__':
    if os.path.exists("./vis"):
        os.system("rm -rf ./vis")
    os.makedirs("./vis")
    for i in range(len(os.listdir('images'))):
        show_annots(i)
