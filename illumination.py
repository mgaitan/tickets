# -*- coding: utf-8 -*-
"""Non-uniform illumination correction

Usage:
  illumination.py <image> [<N>] [<sigma>] [<mu>]

Options:
  -h --help     Show this screen.
"""
import sys

import os

import numpy as np

from docopt import docopt

from scipy.ndimage import sobel

from skimage.io import imread, imsave
from skimage.transform import rescale
from skimage.filter import gaussian_filter


def center(im):
    h, w = im.shape
    h = int(h / 4.)
    w = int(w / 4.)
    return im[h:h+2*h, w:w+2*w]


class NonUniformIllumination(object):
    """" Implementation of the non-uniform illuminaton profile estimation and
    compensation method of Tasdizen et al. The paper has missing details that
    have to be taken care of, eg. the use of weighted least squares instead of
    plain LS to account for the weights matrix.

    To be fixed:

    1. the final estimation of the illumination profile seems to be OK up to a
    multiplicative factor.

    2. the way the illumination is corrected since just dividing by the
    estimated profile doesn't semms work well

    Tasdizen et al. "Non-uniform illumination correction in transmission
    electron microscopy, 2008.

    """

    def __init__(self, N=3, sigma=0.1, mu=10.):
        # model degree
        self.N = int(N)
        if self.N >= 5:
            print('WARNING: there are some numerical issues for N>5')

        # Gaussian smothing
        self.sigma = float(sigma)

        # weights parameter
        self.mu = float(mu)

        self.__eps = 2**-23

        self.profile = None

    def __gradient(self, im):
        im_x = sobel(im, axis=1) / 8.
        im_y = sobel(im, axis=0) / 8.
        return (im_x, im_y)

    def __S(self, x, y):
        if self.N == 1:
            S = [x,  y]
        elif self.N == 2:
            S = [x, y, x*x, x*y, y*y]
        elif self.N == 3:
            S = [x, y, x*x, x*y, y*y, x*x*x, y*x*x, y*y*x, y*y*y]
        else:
            S = []
            for i in range(self.N + 1):
                for j in range(i + 1):
                    S.append(x**(i-j) * y**j)
        return np.column_stack(S[1:])  # remove S[0,0] term

    def __M(self, x, y):
        _1 = np.ones(x.size)
        _0 = np.zeros(x.size)
        if self.N == 1:
            Q = [_1,  _0]
            R = [_0,  _1]
        elif self.N == 2:
            Q = [_1,  _0, 2.*x,  y,   _0]
            R = [_0,  _1,   _0,  x, 2.*y]
        elif self.N == 3:
            Q = [_1, _0, 2.*x, y, _0, 3.*x*x, 2.*x*y, y*y, _0]
            R = [_0, _1, _0, x, 2.*y, _0, x*x, 2.*x*y, 3.*y*y]
        else:
            for i in range(self.N + 1):
                for j in range(i + 1):
                    Q.append((i-j) * x**max(0, i-j-1) * y**j)
                    R.append(x**(i-j) * j*y**max(j-1, 0))
        return np.vstack((np.column_stack(Q[1:]), np.column_stack(R[1:])))

    def fit(self, image):
        if image.ndim > 2:
            raise TypeError('only single channel images allowed')

        # smooth image
        im = gaussian_filter(image, self.sigma)

        # gradients and grad. magnitude
        im_x, im_y = self.__gradient(im)
        im_r = np.sqrt(im_x * im_x + im_y * im_y)

        # log gradients
        log_im = np.log(im + self.__eps)
        log_im_x, log_im_y = self.__gradient(log_im)
        log_im_x = log_im_x.ravel()
        log_im_y = log_im_y.ravel()

        # compute weights
        weights = np.exp(-im_r / (self.mu ** 2))
        weights = weights.ravel()

        # (x,y) coordinates
        ny, nx = im.shape
        x, y = np.meshgrid(range(nx), range(ny), indexing='xy')
        x = x.ravel() - 0.5 * nx
        y = y.ravel() - 0.5 * ny

        # model (gradient) matrix
        M = self.__M(x, y)

        # observed (smoothed) gradient
        g = np.row_stack((log_im_x, log_im_y))
        g.shape = (g.size, 1)

        # diag(weights) * M
        weights.shape = (weights.size, 1)
        WM = M * np.vstack((weights, weights))

        # solve for gamma
        _g = np.asmatrix(g)
        _M = np.asmatrix(M)
        _WM = np.asmatrix(WM)
        gamma = np.linalg.inv(_WM.T * _M) * _WM.T * _g

        # compute illumination profile
        S = self.__S(x, y)
        I = np.asarray(np.asmatrix(S) * gamma)

        self.profile = np.exp(I - I.max())
        self.profile = self.profile.reshape(im.shape)

    def __call__(self, im):
        self.fit(im)
        return im / (1.0 + self.profile)
        #return im / self.profile


def run(imfile, N, sigma, mu):
    N = 2 if N is None else int(N)
    sigma = 1.0 if sigma is None else float(sigma)
    mu = 10.0 if mu is None else float(mu)

    # read image
    im0 = imread(imfile, as_grey=True)

    # rescale to a common size
    scale = 1e6 / float(im0.size)
    im = rescale(im0, (scale, scale))

    # estimate illumination profile
    proc0 = NonUniformIllumination(N=N, sigma=sigma, mu=mu)

    comp = proc0(im)
    illum = proc0.profile

    # # resize to original size
    # illum = rescale(illum, (1.0/scale, 1.0/scale))
    # illum = np.resize(illum, im0.shape)

    fname = os.path.splitext(imfile)

    illum = (illum - illum.min()) / (illum.max() - illum.min())
    imsave(fname[0] + '-illum' + fname[1], illum)

    comp = (comp - comp.min()) / (comp.max() - comp.min())
    imsave(fname[0] + '-comp' + fname[1], comp)

    return

    # thr = threshold_otsu(center(im0))
    # bim = (im > thr)
    # imsave(fname[0] + '-bin' + fname[1], bim.astype(float))

    # thr = threshold_otsu(center(comp))
    # bcomp = (comp > thr)
    # imsave(fname[0] + '-comp-bin' + fname[1], bcomp.astype(float))

if __name__ == "__main__":
    args = docopt(__doc__)
    sys.exit(run(args["<image>"], args["<N>"], args["<sigma>"], args["<mu>"]))
