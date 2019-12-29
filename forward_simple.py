import argparse
import os
import numpy as np
import cv2
import importlib
import svd_xy as sxy
from numpy.fft import fft2, ifft2
from imreg_dft.imreg import *
from numpy.linalg import svd
from scipy.ndimage import correlate, convolve






def main(args):
    stacked = np.load(args.stack_file)
    stack, si_mat = stacked
    weights, weights_interp, comps, u, s, vt = sxy.diffusercam_svd_xy(stack, stack.shape[2]-3, si_mat)
    shape = stack.shape[:2]
    H, b = sxy.make_H(u, shape)
    file_names = os.listdir(args.gt_folder)
    cum_mse = 0
    idx = 0
    for idx, file_name in enumerate(file_names):
        name = args.gt_folder + file_name
        im = sxy.initialize_im(name, shape)
        sim = sxy.forward_rgb(H, im)
        cv2.imwrite(args.save_folder + file_name, sim)
        if args.compare:
            real = sxy.initialize_im(args.diffuser_folder, shape)
            cum_mse += sxy.mse(sim, real)

    if args.compare:
        print("Avg MSE: {}".format(cum_mse/(idx + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='runs forward model on the first n images in a folder')
    parser.add_argument('--gt_folder', type=str, default='~/research/mirflickr25k/diffuser_images_2_14_auto')
    parser.add_argument('--compare', type=bool, default=True)
    parser.add_argument('--stack_file', type=str, default='stacked_psfs_2.npy')
    parser.add_argument('--diffuser_folder', type=str, default='~/research/mirflickr25k_recon/recon_0_iter')
    parser.add_argument('--save_folder', type=str, default='~/research/simulation_results/forward_simple/')
    parser.add_argument('num_images', type=int)
    args = parser.parse_args()
    main(args)
