import os
import sys
import glob
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.ndimage.filters import maximum_filter
from shapely.geometry import Polygon

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import HorizonNet
from dataset import visualize_a_data
from misc import post_proc, panostretch, utils

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator


def find_N_peaks(signal, r=29, min_v=0.05, N=None):
    max_v = maximum_filter(signal, size=r, mode='wrap') 
    pk_loc = np.where(max_v == signal)[0]
    pk_loc = pk_loc[signal[pk_loc] > min_v]
    if N is not None:
        order = np.argsort(-signal[pk_loc])
        pk_loc = pk_loc[order[:N]]
        pk_loc = pk_loc[np.argsort(pk_loc)]

    return pk_loc, signal[pk_loc]


def augment(x_img, flip, rotate):
    x_img = x_img.numpy()
    aug_type = ['']
    x_imgs_augmented = [x_img]
    if flip:
        aug_type.append('flip')
        x_imgs_augmented.append(np.flip(x_img, axis=-1))
    for shift_p in rotate:
        shift = int(round(shift_p * x_img.shape[-1]))
        aug_type.append('rotate %d' % shift)
        x_imgs_augmented.append(np.roll(x_img, shift, axis=-1))
    return torch.FloatTensor(np.concatenate(x_imgs_augmented, 0)), aug_type


def augment_undo(x_imgs_augmented, aug_type):
    x_imgs_augmented = x_imgs_augmented.cpu().numpy()
    sz = x_imgs_augmented.shape[0] // len(aug_type)
    x_imgs = []
    for i, aug in enumerate(aug_type):
        x_img = x_imgs_augmented[i*sz : (i+1)*sz]
        if aug == 'flip':
            x_imgs.append(np.flip(x_img, axis=-1))
        elif aug.startswith('rotate'):
            shift = int(aug.split()[-1])
            x_imgs.append(np.roll(x_img, -shift, axis=-1))
        elif aug == '':
            x_imgs.append(x_img)
        else:
            raise NotImplementedError()

    return np.array(x_imgs)


def inference(net, x, device, ori_width, ori_height, top_pad, bottom_pad, flip=False, rotate=[], visualize=False,
              force_cuboid=False, force_raw=False, min_v=None, r=0.05):
    '''
    net   : the trained HorizonNet
    x     : tensor in shape [1, 3, 512, 1024]
    flip  : fliping testing augmentation
    rotate: horizontal rotation testing augmentation
    '''

    H, W = tuple(x.shape[2:])

    # Network feedforward (with testing augmentation)

    x, aug_type = augment(x, flip, rotate)
    y_bon_, y_cor_ = net(x.to(device))
    y_bon_ = augment_undo(y_bon_.cpu(), aug_type).mean(0)
    y_cor_ = augment_undo(torch.sigmoid(y_cor_).cpu(), aug_type).mean(0)

    # Visualize raw model output
    if visualize:
        vis_out = visualize_a_data(x[0],
                                   torch.FloatTensor(y_bon_[0]),
                                   torch.FloatTensor(y_cor_[0]))
    else:
        vis_out = None

    # y_bon??? ?????? column??? ?????? floor,ceiling ????????? (2,1024)
    # y_cor??? ??? ?????? boundary??? ??????????????? (1024, )
    y_bon_ = (y_bon_[0] / np.pi + 0.5) * H - 0.5
    y_bon_[0] = np.clip(y_bon_[0], 1, H/2-1)
    y_bon_[1] = np.clip(y_bon_[1], H/2+1, H-2)
    y_cor_ = y_cor_[0, 0]

    # Init floor/ceil plane
    z0 = 50
    _, z1 = post_proc.np_refine_by_fix_z(*y_bon_, z0)

    if force_raw:
        # Do not run post-processing, export raw polygon (1024*2 vertices) instead.
        # [TODO] Current post-processing lead to bad results on complex layout.
        cor = np.stack([np.arange(1024), y_bon_[0]], 1)

    else:
        # Detech wall-wall peaks
        if min_v is None:
            min_v = 0 if force_cuboid else 0.05
        r = int(round(W * r / 2))
        N = 4 if force_cuboid else None #???????????? ?????? 4?????? ????????? ??? ??????. default=False
        xs_ = find_N_peaks(y_cor_, r=r, min_v=min_v, N=N)[0] #?????? ??? boundary??? x??????

    # cor_id??? corner??? ?????? ??????(corner??? ??????, 2)
    cor_id = np.zeros((len(xs_)*2, 2), np.float32)
    for i, wall_bon in enumerate(xs_):
        cor_id[i*2] = wall_bon, y_bon_[0, wall_bon] #ceiling??? ????????? ?????? corner index
        cor_id[i*2 + 1] = wall_bon, y_bon_[1, wall_bon] #floor??? ????????? ?????? corner index

    # wall??? bbox ?????????
    crop_width = int(W*0.35)
    crop_cor_id = cor_id[cor_id[:,0]>crop_width] #????????? ?????? ????????? cor_id
    start_points = [[crop_width, crop_cor_id[0,1]], [crop_width, crop_cor_id[1,1]]] # crop?????? ????????? points
    finish_points = [[W,crop_cor_id[-2,1]], [W, crop_cor_id[-1,1]]] # ????????? ?????? points
    new_cor_id = np.concatenate([start_points, crop_cor_id, finish_points]) #start_points??? finish_points??? ????????? cor_id

    # boundary ????????? bbox ?????????(?????? ?????? ????????? 10%??? ???????????? ??????)
    boundary = []
    boundary_1 = np.array([crop_width, top_pad, crop_width + int(W*0.1), bottom_pad]) 
    boundary_2 = np.array([W - int(W*0.1), top_pad, W, bottom_pad])
    boundary.append(boundary_1)
    boundary.append(boundary_2)

    # Normalized to [0, 1]
    new_cor_id[:, 0] /= W
    new_cor_id[:, 1] /= H

    # points????????? bounding box ?????????
    bbox_list = []
    for i in range(0, len(new_cor_id)-2, 2):
        bbox = [new_cor_id[i][0], new_cor_id[i][1], new_cor_id[i+3][0], new_cor_id[i+3][1]]
        bbox_list.append(bbox)


    # bouonding box ?????? ???????????? ????????????
    bbox_list = np.array(bbox_list)
    bbox_list[:,0] *= ori_width
    bbox_list[:,2] *= ori_width
    bbox_list[:,1] *= ori_height
    bbox_list[:,3] *= ori_height
    bbox_list = bbox_list.astype(int)


    # boundary box ?????? ???????????? ????????????
    boundary = np.array(boundary)
    boundary[:,0] = boundary[:,0] * ori_width / W
    boundary[:,1] = boundary[:,1] * ori_width / W
    boundary[:,2] = boundary[:,2] * ori_height / H
    boundary[:,3] = boundary[:,3] * ori_height / H
    boundary = boundary.astype(int)

    return cor_id, z0, z1, vis_out, bbox_list, boundary


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pth', required=True,
                        help='path to load saved checkpoint.')
    parser.add_argument('--img_glob', type=str, required=True,
                        help='NOTE: Remeber to quote your glob path. '
                             'All the given images are assumed to be aligned'
                             'or you should use preporcess.py to do so.')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--visualize', action='store_true')
    # Augmentation related
    parser.add_argument('--flip', action='store_true',
                        help='whether to perfome left-right flip. '
                             '# of input x2.')
    parser.add_argument('--rotate', nargs='*', default=[], type=float,
                        help='whether to perfome horizontal rotate. '
                             'each elements indicate fraction of image width. '
                             '# of input xlen(rotate).')
    # Post-processing realted
    parser.add_argument('--r', default=0.05, type=float)
    parser.add_argument('--min_v', default=None, type=float)
    parser.add_argument('--force_cuboid', action='store_true')
    parser.add_argument('--force_raw', action='store_true')
    # Misc arguments
    parser.add_argument('--no_cuda', action='store_true',
                        help='disable cuda')
    args = parser.parse_args()

    # Prepare image to processed
    # single image
    #paths = sorted(glob.glob(args.img_glob))

    # Multiple
    paths = sorted(glob.glob(args.img_glob+"/*"))

    if len(paths) == 0:
        print('no images found')
    for path in paths:
        assert os.path.isfile(path), '%s not found' % path

    # Check target directory
    if not os.path.isdir(args.output_dir):
        print('Output directory %s not existed. Create one.' % args.output_dir)
        os.makedirs(args.output_dir)
    device = torch.device('cpu' if args.no_cuda else 'cuda')

    # Loaded trained model
    net = utils.load_trained_model(HorizonNet, args.pth).to(device)
    net.eval()

    # Inferencing
    with torch.no_grad():
        for i_path in tqdm(paths, desc='Inferencing'):
            filename = os.path.split(i_path)[-1][:-4]
            image_id = filename[5:]

            # ????????? ????????????
            img_pil = Image.open(i_path)
            w, h = img_pil.size

            # plot ??????
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img_pil)
        

            # (1024, 512)??? resize    
            if img_pil.size != (1024, 512):
                img_pil = img_pil.resize((1024, 512), Image.BICUBIC)
            img_ori = np.array(img_pil)[..., :3].transpose([2, 0, 1]).copy()
            x = torch.FloatTensor([img_ori / 255])

            # ?????? ??????(resize ??????)
            resized_w, resized_h = img_pil.size
            np_img_pil = np.array(img_pil)
            for i in range(resized_h):
                color = np_img_pil[i,-1,:]
                if (color[0] != 0) and (color[1] != 0) and (color[2] != 0):
                    top_padding = i
                    break
            
            for i in reversed(range(resized_h)):
                color = np_img_pil[i,-1,:]
                if (color[0] != 0) and (color[1] != 0) and (color[2] != 0):
                    bottom_padding = i
                    break


            # Inferenceing corners
            cor_id, z0, z1, vis_out, bbox_list, boundary = inference(net=net, x=x, device=device, 
                                                ori_width=w, ori_height = h, 
                                                top_pad = top_padding, bottom_pad = bottom_padding,
                                                flip=args.flip, rotate=args.rotate,
                                                visualize=args.visualize,
                                                force_cuboid=args.force_cuboid,
                                                force_raw=args.force_raw,
                                                min_v=args.min_v, r=args.r)

            #boundary, wall bbox ?????????
            total_bbox = np.concatenate([boundary, bbox_list])
            
            # json ?????? ?????????  
            img = dict()
            
            wall_label = 1001
            for i, (x1, y1, x2, y2) in enumerate(total_bbox):
                obj = dict()
                
                # ?????????????????? boundary(label 999, 1000?????? ??????)
                if i == 0:
                    obj['label'] = 999
                    obj['bbox'] = [int(x1), int(y1), int(x2), int(y2)]
                    obj_index = "boundary_{}".format(i)
                    img[obj_index] = obj

                elif i == 1:
                    obj['label'] = 1000
                    obj['bbox'] = [int(x1), int(y1), int(x2), int(y2)]
                    obj_index = "boundary_{}".format(i)
                    img[obj_index] = obj
                
                # wall??? label 1001~?????? ??????
                else:
                    obj['label'] = wall_label
                    obj['bbox'] = [int(x1), int(y1), int(x2), int(y2)]
                    obj_index = "wall_{}".format(i-2)
                    img[obj_index] = obj
                    wall_label += 1

            # json ?????? ??????
            json_name = image_id + ".json"
            json_path = os.path.join(args.output_dir, f"{filename}.json")

            with open(json_path, 'w', encoding="utf-8") as make_file:
                json.dump(img, make_file, ensure_ascii=False, indent="\t")

            # wall bbox ?????????
            for x1,y1,x2,y2 in bbox_list:
                box_w = x2 - x1
                box_h = y2 - y1

                # Rectangle patch ??????
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor='r', facecolor="none")
                ax.add_patch(bbox)

            # boundary bbox ?????????
            for x1,y1,x2,y2 in boundary:
                box_w = x2 - x1
                box_h = y2 - y1

                # Rectangle patch ??????
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor='b', facecolor="none")
                ax.add_patch(bbox)                

            # filename ?????????
            filename = "bbox_pano_"+i_path.split('_')[-1]
            
            # wall bbox, boundary bbox ????????? ????????? ??????
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            
            image_path = os.path.join(os.getcwd(), 'assets/bbox_img', filename)

            plt.savefig(image_path, bbox_inches="tight", pad_inches=0.0)
            plt.close()

            #Output result
            with open(os.path.join(args.output_dir, filename + '.json'), 'w') as f:
                json.dump({
                    'z0': float(z0),
                    'z1': float(z1),
                    'uv': [[float(u), float(v)] for u, v in cor_id],
                }, f)

            if vis_out is not None:
                vis_path = os.path.join(args.output_dir, filename + '.raw.png')
                vh, vw = vis_out.shape[:2]
                Image.fromarray(vis_out)\
                     .resize((vw//2, vh//2), Image.LANCZOS)\
                     .save(vis_path)
