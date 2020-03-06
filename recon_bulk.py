import sys
sys.path.append('../DiffuserCam_ML/classif/models/')
sys.path.append('../DiffuserCam_ML/classif/')
import argparse
import os
import numpy as np
import cv2
import importlib
import skimage
import skimage.transform
import torch
import models.admm_model as admm_model_plain

from utils import load_psf_image, preplot
from torch.multiprocessing import Pool
from image_utils import *
from tqdm import tqdm
from multiprocessing import set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass




def get_recon(frame, model):
    my_device = 'cuda:0'
    frame_float = frame.astype('float32')#(frame/np.max(frame)).astype('float32')
    perm = torch.tensor(frame_float.transpose((2, 0, 1))).unsqueeze(0)
    with torch.no_grad():
        inputs = perm.to(my_device)
        out = model(inputs)[0].cpu().detach()
    return np.flipud((preplot(out.numpy())*255).astype('uint8'))[...,::-1]


def admm(multi_args):
    file_name, curr_save_folder, path, model = multi_args
    name = os.path.join(path, file_name)
    save_path = os.path.join(curr_save_folder, file_name)
    if os.path.isfile(save_path):
        return False
    # try:
    im = imread_to_normalized_float(name)
    imsave_from_uint8(save_path, get_recon(im, model))
    return True

    # except:
    #     # os.remove(name)
    #     return False



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='runs forward model on the first n images in a folder')
    parser.add_argument('-root', type=str, default='../mirflickr25k/gt_images_2_14_auto/')
    parser.add_argument('-stack_file', type=str, default='stacked_psfs_2.npy')
    parser.add_argument('-diffuser_folder', type=str, default='../mirflickr25k_recon/recon_0_iter/')
    parser.add_argument('-save_folder', type=str, default='../simulation_results/forward_simple/')
    parser.add_argument('-num_images', type=int, default=None)
    parser.add_argument('-multiprocessing_workers', type=int, default=1)
    parser.add_argument('-num_iterations', type=int, default=5)
    parser.add_argument("-psf_file", type=str, default='../recon_files/psf_white_LED_Nick.tiff')
    args = parser.parse_args()

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

    admm_converged2 = admm_model_plain.ADMM_Net(batch_size=1, h=h, iterations=args.num_iterations,
                                                learning_options=learning_options_none, cuda_device=my_device)

    admm_converged2.tau.data = admm_converged2.tau.data * 1000
    admm_converged2.to(my_device)
    model = admm_converged2
    model.share_memory()

    print("Recon Model Created")
    print("Preprocessing")
    pre_pbar = tqdm(total=1000)
    all_files, all_save_folders, all_paths, model_repeated = [], [], [], []
    for path, subdirs, files in os.walk(args.root, topdown=True):

        curr_save_folder = os.path.join(args.save_folder, os.path.relpath(path, args.root))

#        if not curr_save_folder:
#            continue

        if not os.path.isdir(curr_save_folder):
            os.makedirs(curr_save_folder, exist_ok=True)
        
#       print(len(files), curr_save_folder)
        if not files:
            print(len(files), curr_save_folder)
            continue
	new_files = [f for f in files if not os.path.isfile(os.path.join(curr_save_folder, f))]
        all_save_folders.extend([curr_save_folder]*len(new_files))
        all_files.extend(new_files)
        all_paths.extend([path]*len(new_files))
        model_repeated.extend([model]*len(new_files))
        pre_pbar.update(1)
    pre_pbar.close()
    print("Preprocessing Done")
    multiproc_args = list(zip(all_files, all_save_folders, all_paths, model_repeated))
    pbar = tqdm(total=len(all_files))
    p = Pool(processes=args.multiprocessing_workers)
    for res in p.imap_unordered(admm, multiproc_args):  # run_forward, gVars['file_names']):
        pbar.update(1)
    p.close()
    pbar.close()
