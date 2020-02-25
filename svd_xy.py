import numpy as np
import cv2
import time
import os
import matplotlib.pyplot as plt
import imageio
from imreg_dft.imreg import *
from scipy.sparse.linalg import svds
from scipy import interpolate
from numpy.fft import *
from numpy.fft import fft2, ifft2
from imreg_dft.imreg import *
from numpy.linalg import svd
from scipy.ndimage import correlate, convolve

SHAPE=(270, 480)
def diffusercam_svd_xy(stack, rnk, si_mat):
    print("creating matrix")
    Ny, Nx = stack[:, :, 0].shape
    vec = lambda x: x[:].flatten()
    ymat = np.zeros((Ny*Nx, stack.shape[2]))
    for j in range(stack.shape[2]):
        ymat[:, j] = vec(stack[:, :, j])

    print("done")

    print("starting svd")
    tic = time.time()
    u, s, vt = svds(ymat, k=rnk)
    toc = time.time()
    print("SVD took {}".format(tic - toc))

    comps = u.reshape((Ny, Nx, rnk))
    weights = np.zeros((stack.shape[2], rnk))
    for i in range(stack.shape[2]):
        for j in range(rnk):
            weights[i, j] = s[j] * vt[j, i]
    xq = np.arange( -Nx // 2, Nx // 2)
    yq = np.arange(-Ny // 2, Ny // 2)
    Xq, Yq = np.meshgrid(xq, yq)
    weights_interp = np.zeros((Ny, Nx, rnk))
    print("Interpolating")
    for r in range(rnk):
        interpolant_r = interpolate.Rbf(si_mat[1, :].T, si_mat[0, :].T, weights[:, r])
        weights_interp[:, :, r] = np.rot90(interpolant_r(Xq, Yq), 2)
    return weights, weights_interp, comps, u, s, vt

def rescale(img, width=None, height=None):
    if not width and not height:
        raise AssertionError
    if width:
        scale = width/img.shape[1]
        height = int(img.shape[0] * scale)
    else:
        scale = height / img.shape[0]
        width = int(img.shape[1] * scale)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def initialize_im(im_name, shape):
    res = np.zeros(shape + (3,)).astype('uint8')
    rh = res.shape[0]
    rw = res.shape[1]
    img = cv2.imread(im_name)[...,::-1]
    if img.shape[0]/img.shape[1] > shape[0]/shape[1]: #img more vertical than psf
        img = rescale(img, height=shape[0])
        res[:, (rw - img.shape[1])//2:(rw + img.shape[1])//2, :] = img
    else: #img more horizontal than psf
        img = rescale(img, width=shape[1])
        res[(rh - img.shape[0])//2:(rh + img.shape[0])//2, :, :] = img
    return res.astype('float32')#cv2.resize(cv2.imread(im_name)[:, :, ::-1], shape[::-1]).astype('float32')


def imsave(im_name, im):
    imageio.imsave(im_name, (im*255).astype('uint8'))


def register_images(folder, center_name, shape=(270, 480), thresh=45):
    center_im = initialize_im(center_name, shape)#cv2.imread(center_name)
    center_gray = rgb2gray(center_im)
    center_gray[center_gray < thresh] = 0
    file_names = [f for f in os.listdir(folder)]
    res = np.zeros((center_im.shape[0], center_im.shape[1], len(file_names)))
    si_mat = np.zeros((2, len(file_names)))
    for idx, file_name in enumerate(file_names):
        name = folder + file_name
        im = initialize_im(name, shape)#cv2.imread(name)
        im_gray = rgb2gray(im)
        im_gray[im_gray < thresh] = 0
        out = similarity(im_gray, center_gray)
        print("Image Name: ", file_name)
        print(out['tvec'])
        im_trans = transform_img(im, tvec=-out['tvec'])

        imshow(im, center_im, im_trans)
        plt.show()
        res[:, :, idx] = rgb2gray(im_trans)
        si_mat[:, idx] = -out['tvec']
    return res, si_mat

def rgb2gray(img_color):
    return cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

def compress_img(image, k):
    image_reconst_layers = [compress_svd(image[:,:,i],k)[0] for i in range(3)]
    image_reconst = np.zeros(image.shape)
    for i in range(3):
        image_reconst[:,:,i] = image_reconst_layers[i]
    return image_reconst

def compress_svd(mat, k):
    U, s, V = svd(mat, full_matrices=False)
    reconst_matrix = np.dot(U[:,:k],np.dot(np.diag(s[:k]),V[:k,:]))
    return reconst_matrix, s

def test_forward_conv(psf, gt_img, df_img):
    out = correlate(gt_img, psf)
    out = out/np.max(out)
    print("Simulated Model")
    plt.imshow(out)
    plt.show()
    print("Actual Diffuser Img")
    df_img = df_img/np.max(df_img)
    plt.imshow(df_img)
    plt.show()
    squared_error = (out - df_img)*(out - df_img)
    mse = np.mean(squared_error)
    print("MSE is {}".format(mse))

def test_forward_conv_2(psf, gt_img, df_img):
    out = convolve(gt_img, psf)
    print("Simulated Model")
    plt.imshow(out)
    plt.show()
    print("Actual Diffuser Img")
    df_img = df_img/np.max(df_img)
    plt.imshow(df_img)
    plt.show()
    squared_error = (out - df_img)*(out - df_img)
    mse = np.mean(squared_error)
    print("MSE is {}".format(mse))


def forward_svd(H, weights, x):
    print(weights.shape)
    x = x[:,:,0].astype('complex64')
    orig_dim = x.shape
    Y = pad_for_conv(np.zeros(x.shape))
    y_2 = np.zeros(x.shape)
    for r in range(0,H.shape[-1]):
        X = fft2(pad_for_conv(weights[:,:,r]*x))
        Y = Y + X*H[:,:,r]
        y_2 = y_2 + crop(np.real(ifft2(X*H[:,:,r])), x.shape)
        plt.show()
    y = ifft2(Y)
    res =  np.flipud(crop(np.real(y),x.shape)) 
    return res #/np.max(res)

def forward(H, x):
    #     print(fft2(pad_for_conv(x)))
    y = ifft2(fft2(pad_for_conv(x)) * H[:, :, -1])
    res = np.flipud(crop(np.real(y), x.shape))
    return res/np.max(res)

def normalize(im):
    for i in range(im.shape[2]):
        im[:, :, i] = im[:, :, i]/np.max(im[:, :, i])
    return im

def forward_rgb(H, im):
    res = np.zeros(im.shape)
    for i in range(3):
        res[:, :, i] = forward(H, im[:, :, i])
    return res

def mse(im1, im2):
    dims = im1.shape[2]
    mses = []
    for i in range(dims):
        diff = (im1[:, :, i] - im2[:, :, i])
        ch_mse = np.mean(diff*diff)
        # print("che_mse {}".format(ch_mse))
        mses.append(ch_mse)
    return sum(mses)/len(mses)


def crop(im, dim):
    h, w = im.shape
    start_h = h // 2 - dim[0] // 2
    start_w = w // 2 - dim[1] // 2
    if dim[0] % 2 == 0:
        end_h = h // 2 + dim[0] // 2
    else:
        end_h = h // 2 + dim[0] // 2 + 1

    if dim[1] % 2 == 0:
        end_w = w // 2 + dim[1] // 2
    else:
        end_w = w // 2 + dim[1] // 2 + 1

    return im[start_h: end_h, start_w: end_w]


def pad_for_conv(im):
    h, w = im.shape[0], im.shape[1]
    res = np.zeros((2 * h, 2 * w) + im.shape[2:]).astype('complex64')
    if len(im.shape) == 2:
        res[h // 2:-h // 2, w // 2:-w // 2] = im
    else:
        res[h // 2:-h // 2, w // 2:-w // 2, :] = im
    return res


def make_H(u, shape):
    x_black = shape[1]//8
    n = u.shape[1]
    b = np.copy(u)
    b = b.reshape(shape + (n,)).astype('complex64')
    H_new = pad_for_conv(np.zeros(b.shape))
    for i in range(n):
        b_calib = b[:, :, i] - np.max(b[:, :x_black, i])
        b_calib[b_calib < 0] = 0
        H_new[:, :, i] = fft2(fftshift(pad_for_conv(b_calib)))
    return H_new, b_calib
