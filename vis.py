import cv2
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
    img = cv2.line(img, source_px, tuple(imgpts[0].ravel()), (255,0,0), 2)
    img = cv2.line(img, source_px, tuple(imgpts[1].ravel()), (0,255,0), 2)
    img = cv2.line(img, source_px, tuple(imgpts[2].ravel()), (0,0,255), 2)
    return img

def project_3d_point(transformation_matrix,p,render_size):
    p1 = transformation_matrix @ Vector((p.x, p.y, p.z, 1))
    p2 = Vector(((p1.x/p1.w, p1.y/p1.w)))
    p2 = (np.array(p2) - (-1))/(1 - (-1)) # Normalize -1,1 to 0,1 range
    pixel = [int(p2[0] * render_size[0]), int(render_size[1] - p2[1]*render_size[1])]
    return pixel

def show_annots(idx, save=True):
    image_filename = "images/%05d.jpg"%idx
    img = cv2.imread(image_filename)
    world_to_cam = Matrix(np.load('annots/cam_to_world.npy'))
    H,W,C = img.shape
    render_size = (W,H)
    metadata = np.load("annots/%05d.npy"%idx, allow_pickle=True)
    trans = metadata.item().get("trans")
    rot_euler = metadata.item().get("rot")
    pixel = tuple(metadata.item().get("pixel").astype(int))
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
    pixel = (200/60)*metadata.item().get("pixel")
    pixel = tuple(pixel.astype(int))

    vis = img.copy()
    vis = draw(vis,center_projected,axes_projected)
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
