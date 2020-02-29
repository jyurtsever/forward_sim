import sys
sys.path.append('../DiffuserCam_ML/classif/models/')
sys.path.append('../DiffuserCam_ML/classif/')
import argparse
import os
import numpy as np
import cv2
import importlib
import imageio
import svd_xy
import skimage
import torch
import admm_model as admm_model_plain

from utils import load_psf_image, preplot
from svd_xy import *
from imreg_dft.imreg import *
from multiprocessing import Pool
from tqdm import tqdm



# lock = threading.RLock()

def main(args):
    stacked = np.load(args.stack_file, allow_pickle=True)
    stack, si_mat = stacked
    weights, weights_interp, comps, u, s, vt = diffusercam_svd_xy(stack, 15, si_mat)

    gVars['shape'] = stack.shape[:2]
    gVars['H'], b = make_H(u, gVars['shape'])
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

#        num_corrupted = 0
        print(curr_save_folder)

        if args.num_images:
            gVars['file_names'] = gVars['file_names'][:args.num_images]

        gVars['pbar'] = tqdm(total=len(files))

#        p = Pool(processes=args.multiprocessing_workers)
#        for res in p.imap_unordered(admm, gVars['file_names']):   #run_forward, gVars['file_names']):
#            gVars['pbar'].update(1)
#            if not res:
#                num_corrupted += 1
#        p.close()
        for f in gVars['file_names']:
            admm(f)
            gVars['pbar'].update(1)

        finished_folders.append(curr_save_folder)
        np.save('./finished.npy', finished_folders)

#    print(f'{num_corrupted} total files corrupted')

def get_recon(frame):
    frame_float = (frame/np.max(frame)).astype('float32')
    perm = torch.tensor(frame_float.transpose((2, 0, 1))).unsqueeze(0)
    with torch.no_grad():
        inputs = perm.to(my_device)
        out = admm_converged2(inputs)

    return (preplot(out[0].cpu().detach().numpy())*255).astype('uint8')



# def run_forward(file_name):
#     name = os.path.join(gVars['path'], file_name)
#     try:
#         im = initialize_im(name, gVars['shape'])
#         sim = forward_rgb(gVars['H'], im)
#         # gVars['pbar'].update(1)
#         imsave(os.path.join(gVars['curr_save_folder'], file_name), sim)
#         return True
#
#     except:
#         # os.remove(name)
#         return False

def admm(file_name):
    name = os.path.join(gVars['path'], file_name)
    try:
        im = initialize_im(name, gVars['shape'])
        imsave(os.path.join(gVars['curr_save_folder'], file_name), get_recon(im))
        return True

    except:
        # os.remove(name)
        return False



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='runs forward model on the first n images in a folder')
    parser.add_argument('-root', type=str, default='../mirflickr25k/gt_images_2_14_auto/')
    parser.add_argument('-stack_file', type=str, default='stacked_psfs_2.npy')
    parser.add_argument('-diffuser_folder', type=str, default='../mirflickr25k_recon/recon_0_iter/')
    parser.add_argument('-save_folder', type=str, default='../simulation_results/forward_simple/')
    parser.add_argument('-num_images', type=int, default=None)
    parser.add_argument('-multiprocessing_workers', type=int, default=1)
    parser.add_argument("-psf_file", type=str, default='../recon_files/psf_white_LED_Nick.tiff')
    args = parser.parse_args()

    gVars = {}
    print("Creating Recon Model")
    my_device = 'cuda:0'

    psf_diffuser = load_psf_image(args.psf_file, downsample=1, rgb=False)

    ds = 4  # Amount of down-sampling.  Must be set to 4 to use dataset images

    print('The shape of the loaded diffuser is:' + str(psf_diffuser.shape))

    psf_diffuser = np.sum(psf_diffuser, 2)

    h = skimage.transform.resize(psf_diffuser,
                                 (psf_diffuser.shape[0] // ds, psf_diffuser.shape[1] // ds),
                                 mode='constant', anti_aliasing=True)

    var_options = {'plain_admm': [],
                   'mu_and_tau': ['mus', 'tau'],
                   }

    learning_options_none = {'learned_vars': var_options['plain_admm']}

    admm_converged2 = admm_model_plain.ADMM_Net(batch_size=1, h=h, iterations=100,
                                                learning_options=learning_options_none, cuda_device=my_device)

    admm_converged2.tau.data = admm_converged2.tau.data * 1000
    admm_converged2.to(my_device)
    print("Recon Model Created")
    main(args)
