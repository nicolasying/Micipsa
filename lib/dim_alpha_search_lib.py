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
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.model_selection import KFold, LeaveOneGroupOut


def parallel_fit(alpha, x_train, y_train, x_test, y_test):
    model = Ridge(alpha=alpha, fit_intercept=False,
                  normalize=False, copy_X=True).fit(x_train, y_train)
    # print(r2_score(y_train, model.predict(x_train)), r2_score(y_test, model.predict(x_test)))
    return r2_score(y_test, model.predict(x_test), multioutput='raw_values')
        # , \r2_score(y_train, model.predict(x_train), multioutput='raw_values'), 
        # explained_variance_score(y_train, model.predict(x_train), multioutput='raw_values'), explained_variance_score(y_test, model.predict(x_test), multioutput='raw_values')

def dim_alpha_search_with_log(fmri_runs, design_matrices, alphas, dimensions, loglabel, model, output_dir, send_mail_log, core_number, verbose):
    n_alpha = len(alphas)
    n_dim = len(dimensions)
    n_train = len(fmri_runs)
    n_voxel = fmri_runs[0].shape[1]

    train = [i for i in range(0, n_train)]
    # r2_cv_train_score = np.zeros((n_dim, n_alpha, n_voxel), dtype=np.float64)
    r2_cv_test_score = np.zeros((n_dim, n_alpha, n_voxel), dtype=np.float64)
    
    for idx, cv_test_id in enumerate(train):
        score_file_name = op.join(
            output_dir, 'cache', "{}_fold_{}.npz".format(loglabel, idx))
        search_name =  op.join(
            output_dir, 'cache', "*{}*_fold_{}.npz".format(loglabel, idx))
        file_list = glob.glob(search_name)
        if len(file_list) > 0:
            print('Fold {}/{} for {} of {} exists.'.format(idx, n_train, loglabel, model), flush=True)
            continue
        print('Fold {}/{}'.format(idx, n_train), flush=True)
        fmri_data = np.vstack([fmri_runs[i] for i in train if i != cv_test_id])
        predictors_ref = np.vstack([design_matrices[i]
                                    for i in train if i != cv_test_id])

        # n_images_train = predictors_ref.shape[0]
        # n_images_test = design_matrices[cv_test_id].shape[0]

        parallel_res = Parallel(n_jobs=core_number, prefer="threads")(
            delayed(parallel_fit)(alpha, predictors_ref[:, :dim], fmri_data, \
                design_matrices[cv_test_id][:, :dim], fmri_runs[cv_test_id])
            for idx1, dim in enumerate(dimensions) for idx2, alpha in enumerate(alphas))

        # parallel_res = np.array(parallel_res).reshape(
        #     n_dim, n_alpha, n_voxel)
        # r2_cv_train_score = parallel_res
        r2_cv_test_score = np.array(parallel_res).reshape(n_dim, n_alpha, n_voxel)
        # r2_cv_test_score = parallel_res[:, :, 1, :]
        # mse_cv_train_score = parallel_res[:, :, 2, :]
        # mse_cv_test_score = parallel_res[:, :, 3, :]

        np.savez_compressed(score_file_name, r2_test=r2_cv_test_score,  \
            #  mse_test=mse_cv_test_score, mse_train=mse_cv_train_score,  \r2_train=r2_cv_train_score,
                 alpha=np.array(alphas), dimension=np.array(dimensions))
                 #ev_test=ev_cv_test_score, ev_train=ev_cv_train_score,
        # r2_cv_train_score.save(train_file_name)
        # r2_cv_test_score.save(test_file_name)

        # files.download(log_file_name)
        if verbose:
            msg = 'Fold {}/{} of subject {} dumped'.format(
                idx, n_train, loglabel)
            send_mail_log('{} loop'.format(model), msg)

    return
