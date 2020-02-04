import argparse
import os
import numpy as np
import cv2
import importlib
import imageio
from svd_xy import *
from numpy.fft import fft2, ifft2
from imreg_dft.imreg import *
from numpy.linalg import svd
from scipy.ndimage import correlate, convolve
import scipy.misc as scm
from progressbar import ProgressBar





def main(args):
    stacked = np.load(args.stack_file, allow_pickle=True)
    stack, si_mat = stacked
    weights, weights_interp, comps, u, s, vt = diffusercam_svd_xy(stack, 15, si_mat)
    shape = stack.shape[:2]
    H, b = make_H(u, shape)
    file_names_gt = os.listdir(args.gt_folder)
    file_names_diffuser = os.listdir(args.diffuser_folder)
    file_names = list(set(file_names_gt) & set(file_names_diffuser))
    if args.num_images:
        file_names = file_names[:args.num_images]

    cum_mse = 0
    idx = 0
    pbar = ProgressBar().start()
    for idx, file_name in enumerate(file_names):
        name = args.gt_folder + file_name
        im = initialize_im(name, shape)
        sim = forward_rgb(H, im)
        pbar.update(idx/len(file_names)*100)
        imsave(args.save_folder + file_name, sim)
        # plt.imshow(sim)
        # plt.show()
        if args.compare:
            real = initialize_im(args.diffuser_folder + file_name, shape)
            curr_mse = mse(sim, normalize(real))
            # plt.imshow(normalize(real))
            # plt.show()
            cum_mse += curr_mse

    if args.compare:
        print("Avg MSE: {}".format(cum_mse/(idx + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='runs forward model on the first n images in a folder')
    parser.add_argument('--gt_folder', type=str, default='../mirflickr25k/gt_images_2_14_auto/')
    parser.add_argument('--compare', type=bool, default=True)
    parser.add_argument('--stack_file', type=str, default='stacked_psfs_2.npy')
    parser.add_argument('--diffuser_folder', type=str, default='../mirflickr25k_recon/recon_0_iter/')
    parser.add_argument('--save_folder', type=str, default='../simulation_results/forward_simple/')
    parser.add_argument('--num_images', type=int, default = None)
    args = parser.parse_args()
    main(args)
