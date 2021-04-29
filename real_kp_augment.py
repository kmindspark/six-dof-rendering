import os
import xml.etree.cElementTree as ET
from xml.dom import minidom
import random
import colorsys
import numpy as np
from cv2 import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import matplotlib.pyplot as plt

sometimes = lambda aug: iaa.Sometimes(0.3, aug)

AUGS = [
    iaa.LinearContrast((0.95, 1.05), per_channel=0.25),
    iaa.Add((-10, 10), per_channel=False),
    iaa.GammaContrast((0.85, 1.15)),
    #iaa.GaussianBlur(sigma=(0.5, 0.8)),
    # iaa.GaussianBlur(sigma=(1,2)),
    iaa.MultiplySaturation((0.95, 1.05)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.0125*255)),
    sometimes(
        iaa.CropAndPad(
        percent=(0.1, 0.2),
        pad_mode=["constant"],
        pad_cval=(0, 20)
    ))
    ]

seq = iaa.Sequential(AUGS, random_order=True)
# blur = iaa.Sequential([iaa.GaussianBlur(sigma=(1,2))], random_order=True)

def augment(input, annots, output_dir_img, output_annot_dir, new_idx, show=False):
    width, height, channels = input.shape
    img = input if channels == 3 else input[:, :, :3]
    img = img.astype(np.uint8)
    # img_blur = blur(image=img)
    img_aug, kpts = seq(image=img, keypoints=np.array([[annots.item().get("trans")[:2], annots.item().get("pixel")]]))
    if show:
        cv2.imshow("img", img_aug)
        cv2.waitKey(0)

    if channels == 3:
        cv2.imwrite(os.path.join(output_dir_img, "%05d.jpg"%new_idx), img_aug)
    else:
        gauss = input[:, :, 3].reshape((width, height, 1))
        combined = np.append(img_aug, gauss, axis=2)
        np.save(os.path.join(output_dir_img, "%05d.npy"%new_idx), combined)
    annots.item()["pixel"] = np.squeeze(kpts)[1]
    np.save(os.path.join(output_annot_dir, '%05d.npy'%new_idx), annots)


if __name__ == '__main__':
    img_dir = 'images'
    annots_dir = 'annots'
    if os.path.exists("./aug"):
        os.system("rm -rf ./aug")
    os.makedirs("./aug")
    os.makedirs("./aug/images")
    os.makedirs("./aug/annots")
    output_dir_img = "aug/images"
    output_annot_dir = "aug/annots"
    os.system("cp annots/cam_to_world.npy %s/"%output_annot_dir)
    idx = len(os.listdir(img_dir))
    orig_len = len(os.listdir(img_dir))
    num_augs_per = 7
    new_idx = 0
    for i in range(orig_len):
        print(i, orig_len)
        img = cv2.imread(os.path.join(img_dir, '%05d.jpg'%i))
        #img = np.load(os.path.join(img_dir, '%05d.npy'%i), allow_pickle=True)
        annots = np.load(os.path.join(annots_dir, '%05d.npy'%i), allow_pickle=True)
        cv2.imwrite(os.path.join(output_dir_img, "%05d.jpg"%new_idx), img)
        #np.save(os.path.join(output_dir_img, '%05d.npy'%new_idx), img)
        np.save(os.path.join(output_annot_dir, '%05d.npy'%new_idx), annots)
        new_idx += 1
        for _ in range(num_augs_per):
            augment(img, annots, output_dir_img, output_annot_dir, new_idx, show=False)
            new_idx += 1
            idx += 1
        idx -= 1
