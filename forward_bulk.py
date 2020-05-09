import argparse
import os
import numpy as np
import cv2
import importlib
from image_utils import *
from imreg_dft.imreg import *
from multiprocessing import Pool
from scipy.signal import convolve2d, fftconvolve
from tqdm import tqdm

def main(args):
    psf = imread_to_normalized_float(args.psf_file)
    psf = rescale(psf, height=psf.shape[0] // args.ds)
    gVars['shape'] = psf.shape[:2]
    gVars['psf'] = psf
    if os.path.isfile('./finished.npy'):
        finished_folders = list(np.load('./finished.npy'))
    else:
        finished_folders = []

    for path, subdirs, files in os.walk(args.root, topdown=True):
        curr_save_folder = os.path.join(args.save_folder, os.path.relpath(path, args.root))
        if curr_save_folder in finished_folders:
            continue

        if not curr_save_folder:
            continue

        if not os.path.isdir(curr_save_folder):
            os.makedirs(curr_save_folder, exist_ok=True)

        if not files:
            continue

        gVars['curr_save_folder'] = curr_save_folder
        gVars['file_names'] = files
        gVars['num_files'] = len(files)
        gVars['path'] = path

        num_corrupted = 0
        print(curr_save_folder)

        if args.num_images:
            gVars['file_names'] = gVars['file_names'][:args.num_images]

        gVars['pbar'] = tqdm(total=len(files))


        p = Pool(processes=args.multiprocessing_workers)
        for res in p.imap_unordered(run_forward, gVars['file_names']):
            gVars['pbar'].update(1)
            if not res:
                num_corrupted += 1
        p.close()
        gVars['pbar'].close()
        finished_folders.append(curr_save_folder)
        np.save('./finished.npy', finished_folders)

    print(f'{num_corrupted} total files corrupted')


"""Basic Fourier Convolution forward model GIVEN TWO NORMALIZED inputs for im and psf"""
def basic_forward(im, psf):
    res = np.zeros(im.shape)
    for i in range(3):
        res[:, :, i] = fftconvolve(im[:, :, i], psf[:, :, i], mode='same')
    return res/np.max(res)


"""Saves results of forward model given file name of input image"""
def run_forward(file_name):
    try:
        name = os.path.join(gVars['path'], file_name)
        im = imread_and_resize_to_normalized_float(name, gVars['shape'])
        sim = basic_forward(im, gVars['psf'])
        imsave_from_normalized_float(os.path.join(gVars['curr_save_folder'], file_name), sim)
    except:
        print(f'There was a problem with image: {name}')


# def run_forward(file_name):
#     name = os.path.join(gVars['path'], file_name)
#     try:
#         im = initialize_im(name, gVars['shape'])
#         sim = forward_rgb(gVars['H'], im)
#     #gVars['pbar'].update(1)
#         imsave(os.path.join(gVars['curr_save_folder'], file_name), sim)
#         return True
#
#     except:
#         # os.remove(name)
#         return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='runs forward model on the first n images in a folder')
    parser.add_argument('-root', type=str, default='../mirflickr25k/gt_images_2_14_auto/')
    # parser.add_argument('-stack_file', type=str, default='stacked_psfs_2.npy')
    parser.add_argument('-save_folder', type=str, default='../simulation_results/forward_simple/')
    parser.add_argument("-psf_file", type=str, default='../recon_files/psf_white_LED_Nick.tiff')
    parser.add_argument("-ds", type=int, default=4)
    parser.add_argument('-num_images', type=int, default = None)
    parser.add_argument('-multiprocessing_workers', type=int, default=6)
    args = parser.parse_args()

    gVars = {}
    main(args)

