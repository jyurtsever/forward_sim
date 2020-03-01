import sys
sys.path.append('../DiffuserCam_ML/classif/models/')
sys.path.append('../DiffuserCam_ML/classif/')
import argparse
import os
import numpy as np
import cv2
import importlib
import skimage
import torch
import admm_model as admm_model_plain
import torch.multiprocessing as tm
from utils import load_psf_image, preplot
from torch.multiprocessing import Pool, Process, set_start_method
from image_utils import *
from tqdm import tqdm

torch.multiprocessing.set_start_method('spawn', force=True)

gVars = {}
# lock = threading.RLock()


    # except:
    #     # os.remove(name)
    #     return false


def get_recon(frame, model):
    my_device = 'cuda:0'
    frame_float = frame.astype('float32')#(frame/np.max(frame)).astype('float32')
    perm = torch.tensor(frame_float.transpose((2, 0, 1))).unsqueeze(0)
    with torch.no_grad():
        inputs = perm.to(my_device)
        out = model(inputs)
    return np.flipud((preplot(out[0].cpu().detach().numpy())*255).astype('uint8'))[...,::-1]


def admm(args):
    file_name, curr_save_folder, path, model = args
    name = os.path.join(path, file_name)
    # try:
    #print(name)
    im = imread_to_normalized_float(name)
    imsave_from_uint8(os.path.join(curr_save_folder, file_name), get_recon(im, model))
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='runs forward model on the first n images in a folder')
    parser.add_argument('-root', type=str, default='../mirflickr25k/gt_images_2_14_auto/')
    parser.add_argument('-stack_file', type=str, default='stacked_psfs_2.npy')
    parser.add_argument('-diffuser_folder', type=str, default='../mirflickr25k_recon/recon_0_iter/')
    parser.add_argument('-save_folder', type=str, default='../simulation_results/forward_simple/')
    parser.add_argument('-num_images', type=int, default=None)
    parser.add_argument('-multiprocessing_workers', type=int, default=1)
    parser.add_argument('-num_iterations', type=int, default=10)
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
    admm_converged2.share_memory()
    print("Recon Model Created")
    
    if os.path.isfile('./finished_recon.npy'):
        finished_folders = list(np.load('./finished_recon.npy'))
    else:
        finished_folders = []

    all_files = []
    all_save_folders = []
    all_paths = []
    all_models = []
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
        all_files.extend(files)
        all_save_folders.extend([curr_save_folder]*len(files))
        all_paths.extend([path]*len(files))
        all_models.extent([admm_converged2]*len(files))

#        finished_folders.append(curr_save_folder)
        #np.save('./finished_recon.npy', finished_folders)
    
    gVars['pbar'] = tqdm(total=len(files))

    multi_args = list(zip(all_files, all_save_folders, all_paths, all_models))
    #print(multi_args)

    p = tm.Pool(processes=args.multiprocessing_workers)
    for res in p.imap_unordered(admm, multi_args):   #run_forward, gVars['file_names']):
        gVars['pbar'].update(1)
    p.close()

    gVars['pbar'].close()
