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
from multiprocessing import Pool
import scipy.misc as scm
from progressbar import ProgressBar



def main(args):
    stacked = np.load(args.stack_file, allow_pickle=True)
    stack, si_mat = stacked
    weights, weights_interp, comps, u, s, vt = diffusercam_svd_xy(stack, 15, si_mat)

    gVars['shape'] = stack.shape[:2]
    gVars['H'], b = make_H(u, gVars['shape'])
    file_names_gt = os.listdir(args.gt_folder)
    file_names_diffuser = os.listdir(args.diffuser_folder)
    print(file_names_gt)
    if args.compare:
        print("yoo")
        gVars['file_names'] = list(set(file_names_gt) & set(file_names_diffuser))
    else:
        gVars['file_names'] = file_names_gt

    gVars['num_files'] = len(gVars['file_names'])

    gVars['idx'], gVars['cum_mse'] = 0, 0

    if args.num_images:
        gVars['file_names'] = gVars['file_names'][:args.num_images]

    gVars['pbar'] = ProgressBar().start()
    print(len(gVars['file_names']))

    with Pool(processes=args.multiprocessing_workers) as p:
        p.map(run_forward, gVars['file_names'])
    # for idx, file_name in enumerate(file_names):
    #     name = args.gt_folder + file_name
    #     im = initialize_im(name, shape)
    #     sim = forward_rgb(H, im)
    #     pbar.update(idx/len(file_names)*100)
    #     imsave(args.save_folder + file_name, sim)
    #     # plt.imshow(sim)
    #     # plt.show()
    #     if args.compare:
    #         real = initialize_im(args.diffuser_folder + file_name, shape)
    #         curr_mse = mse(sim, normalize(real))
    #         # plt.imshow(normalize(real))
    #         # plt.show()
    #         cum_mse += curr_mse

    if args.compare:
        print("Avg MSE: {}".format(gVars['cum_mse']/(gVars['idx'] + 1)))



def run_forward(file_name):
    name = args.gt_folder + file_name
    im = initialize_im(name, gVars['shape'])
    sim = forward_rgb(gVars.H, im)
    gVars['pbar'].update(gVars['idx']/ len(gVars['num_files']) * 100)
    imsave(args.save_folder + file_name, sim)
    # plt.imshow(sim)
    # plt.show()
    if args.compare:
        real = initialize_im(args.diffuser_folder + file_name, gVars['shape'])
        curr_mse = mse(sim, normalize(real))
        # plt.imshow(normalize(real))
        # plt.show()
        gVars['cum_mse'] += curr_mse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='runs forward model on the first n images in a folder')
    parser.add_argument('-gt_folder', type=str, default='../mirflickr25k/gt_images_2_14_auto/')
    parser.add_argument('-compare', type=bool, default=False)
    parser.add_argument('-stack_file', type=str, default='stacked_psfs_2.npy')
    parser.add_argument('-diffuser_folder', type=str, default='../mirflickr25k_recon/recon_0_iter/')
    parser.add_argument('-save_folder', type=str, default='../simulation_results/forward_simple/')
    parser.add_argument('-num_images', type=int, default = None)
    parser.add_argument('-multiprocessing_workers', type=int, default=6)
    args = parser.parse_args()

    gVars = {}
    main(args)


