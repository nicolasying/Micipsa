#cython: language_level=3, boundscheck=False
# IMPORTS
import csv
import getopt
import glob
import os
import os.path as op
import pickle
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
from joblib import Parallel, delayed, dump, load
from nilearn.image import coord_transform, math_img, mean_img, threshold_img
from nilearn.input_data import MultiNiftiMasker
from nilearn.masking import apply_mask, compute_epi_mask
from nilearn.plotting import plot_glass_brain
from numpy.random import randint
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut

cimport numpy as np


def clean_rscorer(estimator, np.ndarray[double, ndim=2] X, np.ndarray[float, ndim=2] y):
    cdef np.ndarray[double, ndim = 1] x1
    x1 = r2_score(y, estimator.predict(X), multioutput='raw_values')
    x1 = np.array([0 if (x < .0 or x >= .99) else x for x in x1])
    return x1


def parallel_fit(double alpha, np.ndarray[double, ndim=2] x_train, np.ndarray[float, ndim=2] y_train, np.ndarray[double, ndim=2] x_test, np.ndarray[float, ndim=2] y_test):
    model = Ridge(alpha=alpha, fit_intercept=False,
                  normalize=False).fit(x_train, y_train)
    # print(r2_score(y_train, model.predict(x_train)), r2_score(y_test, model.predict(x_test)))
    return clean_rscorer(model, x_train, y_train), clean_rscorer(model, x_test, y_test)


def dim_alpha_search(fmri_runs, design_matrices, alphas, dimensions, loglabel, dump_file):
    pickle.dump(loglabel, dump_file, protocol=4)
    pickle.dump(alphas, dump_file, protocol=4)
    pickle.dump(dimensions, dump_file, protocol=4)

    cdef int n_alpha = len(alphas)
    cdef int n_dim = len(dimensions)
    cdef int n_train = len(fmri_runs)
    cdef int n_voxel = fmri_runs[0].shape[1]
    cdef np.ndarray[double, ndim = 4] r2_cv_train_score
    cdef np.ndarray[double, ndim = 4] r2_cv_test_score

    train = [i for i in range(0, n_train)]
    r2_cv_train_score = np.zeros(
        (n_train, n_dim, n_alpha, n_voxel), dtype=np.float64)
    r2_cv_test_score = np.zeros(
        (n_train, n_dim, n_alpha, n_voxel), dtype=np.float64)

    for idx, cv_test_id in enumerate(train):
        fmri_data = np.vstack([fmri_runs[i] for i in train if i != cv_test_id])
        predictors_ref = np.vstack([design_matrices[i]
                                    for i in train if i != cv_test_id])
        n_images_train = predictors_ref.shape[0]
        n_images_test = design_matrices[cv_test_id].shape[0]
        parallel_res = Parallel(n_jobs=-2, prefer="threads")(
            delayed(parallel_fit)(
                alpha, predictors_ref[:, :dim], fmri_data, design_matrices[cv_test_id][:, :dim], fmri_runs[cv_test_id])
            for idx1, dim in enumerate(dimensions) for idx2, alpha in enumerate(alphas))

        parallel_res = np.array(parallel_res).reshape(
            n_dim, n_alpha, 2, n_voxel)
        r2_cv_train_score[idx, :, :, :] = parallel_res[:, :, 0, :]
        r2_cv_test_score[idx, :, :, :] = parallel_res[:, :, 1, :]

    pickle.dump(r2_cv_train_score, dump_file, protocol=4)
    pickle.dump(r2_cv_test_score, dump_file, protocol=4)

    return r2_cv_train_score.mean(axis=0).max(axis=(0, 1)), r2_cv_test_score.mean(axis=0).max(axis=(0, 1))


def dim_alpha_search_with_log(fmri_runs, design_matrices, alphas, dimensions, loglabel, model, output_dir, send_mail_log, core_number=-1):

    cdef int n_alpha = len(alphas)
    cdef int n_dim = len(dimensions)
    cdef int n_train = len(fmri_runs)
    cdef int n_voxel = fmri_runs[0].shape[1]
    cdef np.ndarray[double, ndim = 3] r2_cv_train_score
    cdef np.ndarray[double, ndim = 3] r2_cv_test_score

    train = [i for i in range(0, n_train)]
    r2_cv_train_score = np.zeros((n_dim, n_alpha, n_voxel), dtype=np.float64)
    r2_cv_test_score = np.zeros((n_dim, n_alpha, n_voxel), dtype=np.float64)

    for idx, cv_test_id in enumerate(train):
        log_file_name = op.join(
            output_dir, 'cache', "{}_run_dim_alpha_search_fold_{}.pkl".format(loglabel, idx))
        search_name =  op.join(
            output_dir, 'cache', "*{}_run_dim_alpha_search_fold_{}.pkl".format(loglabel, idx))
        file_list = glob.glob(search_name)
        if len(file_list) > 0:
            print('Fold {}/{} for {} of {} exists.'.format(idx, n_train, loglabel, model), flush=True)
            continue

        print('Fold {}/{}'.format(idx, n_train), flush=True)
        fmri_data = np.vstack([fmri_runs[i] for i in train if i != cv_test_id])
        predictors_ref = np.vstack([design_matrices[i]
                                    for i in train if i != cv_test_id])
        n_images_train = predictors_ref.shape[0]
        n_images_test = design_matrices[cv_test_id].shape[0]

        parallel_res = Parallel(n_jobs=core_number, prefer="threads")(
            delayed(parallel_fit)(alpha, predictors_ref[:, :dim], fmri_data, \
                design_matrices[cv_test_id][:, :dim], fmri_runs[cv_test_id])
            for idx1, dim in enumerate(dimensions) for idx2, alpha in enumerate(alphas))

        parallel_res = np.array(parallel_res).reshape(
            n_dim, n_alpha, 2, n_voxel)
        r2_cv_train_score = parallel_res[:, :, 0, :]
        r2_cv_test_score = parallel_res[:, :, 1, :]
        with open(log_file_name, "ab+") as dump_file:
            pickle.dump(loglabel, dump_file, protocol=4)
            pickle.dump(alphas, dump_file, protocol=4)
            pickle.dump(dimensions, dump_file, protocol=4)
            pickle.dump(r2_cv_train_score, dump_file, protocol=4)
            pickle.dump(r2_cv_test_score, dump_file, protocol=4)

        # files.download(log_file_name)
        msg = 'Fold {}/{} of subject {} dumped'.format(
            idx, n_train, loglabel)
        send_mail_log('{} loop'.format(model), msg)

    return
