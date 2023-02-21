# -*- encoding: utf-8 -*-
'''
@File    :   prepare_data.py
@Time    :   2023/02/21 10:55:57
@Author  :   sdu-huanhq
@Version :   1.0
@Contact :   779613975@qq.com
@Reference : Pvnet(cvpr2019)
'''

import os
import argparse
import numpy as np
import json
import tqdm
from PIL import Image
from utils import *

def record_occ_ann(model_meta, img_id, ann_id, images, annotations):
    data_root = 'data/occlusion_linemod'
    model_meta['data_root'] = data_root
    cls = model_meta['cls']
    split = model_meta['split']
    corner_3d = model_meta['corner_3d']
    center_3d = model_meta['center_3d']
    fps_3d = model_meta['fps_3d']
    K = model_meta['K']

    inds = np.loadtxt(os.path.join('data/linemod', cls, 'test_occlusion.txt'), np.str)
    inds = [int(os.path.basename(ind).replace('.jpg', '')) for ind in inds]

    rgb_dir = os.path.join(data_root, 'RGB-D/rgb_noseg')
    for ind in tqdm.tqdm(inds):
        img_name = 'color_{:05d}.png'.format(ind)
        rgb_path = os.path.join(rgb_dir, img_name)
        pose_dir = os.path.join(data_root, 'blender_poses', cls)
        pose_path = os.path.join(pose_dir, 'pose{}.npy'.format(ind))
        if not os.path.exists(pose_path):
            continue

        rgb = Image.open(rgb_path)
        img_size = rgb.size
        img_id += 1
        info = {'file_name': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': img_id}
        images.append(info)

        pose = np.load(pose_path)

        corner_2d = project(corner_3d, K, pose)
        center_2d = project(center_3d[None], K, pose)[0]
        fps_2d = project(fps_3d, K, pose)

        mask_path = os.path.join(data_root, 'masks', cls, '{}.png'.format(ind))
        depth_path = os.path.join(data_root, 'RGB-D', 'depth_noseg',
                'depth_{:05d}.png'.format(ind))

        ann_id += 1
        anno = {'mask_path': mask_path, 'depth_path': depth_path,
                'image_id': img_id, 'category_id': 1, 'id': ann_id}
        anno.update({'corner_3d': corner_3d.tolist(), 'corner_2d': corner_2d.tolist()})
        anno.update({'center_3d': center_3d.tolist(), 'center_2d': center_2d.tolist()})
        anno.update({'fps_3d': fps_3d.tolist(), 'fps_2d': fps_2d.tolist()})
        anno.update({'K': K.tolist(), 'pose': pose.tolist()})
        anno.update({'data_root': rgb_dir})
        anno.update({'type': 'render', 'cls': cls})
        annotations.append(anno)

    return img_id, ann_id

def record_real_ann(model_meta, img_id, ann_id, images, annotations):
    data_root = model_meta['data_root']
    cls = model_meta['cls']
    split = model_meta['split']
    corner_3d = model_meta['corner_3d']
    center_3d = model_meta['center_3d']
    fps_3d = model_meta['fps_3d']
    K = model_meta['K']

    pose_dir = os.path.join(data_root, cls, 'pose')
    rgb_dir = os.path.join(data_root, cls, 'JPEGImages')

    inds = np.loadtxt(os.path.join(data_root, cls, split + '.txt'), np.str)
    inds = [int(os.path.basename(ind).replace('.jpg', '')) for ind in inds]

    for ind in tqdm.tqdm(inds):
        img_name = '{:06}.jpg'.format(ind)
        rgb_path = os.path.join(rgb_dir, img_name)
        rgb = Image.open(rgb_path)
        img_size = rgb.size
        img_id += 1
        info = {'file_name': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': img_id}
        images.append(info)

        pose_path = os.path.join(pose_dir, 'pose{}.npy'.format(ind))
        pose = np.load(pose_path)
        corner_2d = project(corner_3d, K, pose)
        center_2d = project(center_3d[None], K, pose)[0]
        fps_2d = project(fps_3d, K, pose)

        mask_path = os.path.join(data_root, cls, 'mask', '{:04d}.png'.format(ind))

        ann_id += 1
        depth_path = os.path.join('data/linemod_orig', cls, 'data', 'depth{}.dpt'.format(ind))
        anno = {'mask_path': mask_path, 'depth_path': depth_path,
                'image_id': img_id, 'category_id': 1, 'id': ann_id}
        anno.update({'corner_3d': corner_3d.tolist(), 'corner_2d': corner_2d.tolist()})
        anno.update({'center_3d': center_3d.tolist(), 'center_2d': center_2d.tolist()})
        anno.update({'fps_3d': fps_3d.tolist(), 'fps_2d': fps_2d.tolist()})
        anno.update({'K': K.tolist(), 'pose': pose.tolist()})
        anno.update({'data_root': rgb_dir})
        anno.update({'type': 'real', 'cls': cls})
        annotations.append(anno)

    return img_id, ann_id


linemod_K = np.array([[572.4114, 0., 325.2611],
                  [0., 573.57043, 242.04899],
                  [0., 0., 1.]])


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--cls", type=str, default="duck")
    parser.add_argument("--split", type=str, default="occ")
    args = parser.parse_args()

    data_root = 'data/linemod'
    model_path = os.path.join(data_root, args.cls, args.cls+'.ply')

    renderer = OpenGLRenderer(model_path)
    K = linemod_K #! linemod intrinsic

    model = renderer.model['pts'] / 1000
    corner_3d = get_model_corners(model)
    center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
    fps_3d = np.loadtxt(os.path.join(data_root, args.cls, 'farthest.txt'))

    model_meta = {
        'K': K, #! camera intrinsic
        'corner_3d': corner_3d, #! 3d bounding box
        'center_3d': center_3d, #! 3d center
        'fps_3d': fps_3d, #! fast point sampling 3D points
        'data_root': data_root,
        'cls': args.cls,
        'split': args.split #! train, test, occ
    }

    img_id = 0
    ann_id = 0
    images = []
    annotations = []

    if args.split == 'occ':
        img_id, ann_id = record_occ_ann(model_meta, img_id, ann_id, images, annotations)

    elif args.split == 'train' or args.split == 'test':
        img_id, ann_id = record_real_ann(model_meta, img_id, ann_id, images, annotations)

    categories = [{'supercategory': 'none', 'id': 1, 'name': args.cls}]
    instance = {'images': images, 'annotations': annotations, 'categories': categories}

    anno_path = os.path.join(data_root, args.cls, args.split + '.json')
    with open(anno_path, 'w') as f:
        json.dump(instance, f)
