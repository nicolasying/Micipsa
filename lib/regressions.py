import csv
import gc
import getopt
import glob
import os
import os.path as op
import pickle
import shutil
import smtplib
import ssl
import sys
import time
import warnings

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load
from nilearn.input_data import MultiNiftiMasker
from nilearn.masking import apply_mask, compute_multi_epi_mask
from nilearn.image import coord_transform, math_img, mean_img, threshold_img
from numpy.random import randint
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from .dim_alpha_search_lib import dim_alpha_search_with_log
from .notifyer import send_mail_log


def generate_group_imgs(subject_list, input_dir, output_dir, file_id='test', result_type='r2'):
    if isinstance(result_type, list):
        for ty in result_type:
            generate_group_imgs(subject_list=subject_list, input_dir=input_dir, output_dir=output_dir, file_id=file_id, result_type=ty)
        return

    if not op.isdir(output_dir):
        os.makedirs(output_dir)

    files = glob.glob(op.join(input_dir, file_id+'*_'+result_type+'.nii.gz'))
    if len(subject_list) != len(files):
        print("Not all {}/{} subjects have its first_level_analysis results.".format(len(files), len(subject_list)))
        return
    image_search = glob.glob(op.join(output_dir, file_id + '*_group_{}.nii.gz'.format(result_type)))
    if len(image_search) > 0:
        print('Group result for', file_id, 'hit cache:', len(image_search))
        return nib.load(image_search[-1])
    group_img = mean_img(files)
    nib.save(group_img, op.join(output_dir, file_id + '_group_{}.nii.gz'.format(result_type)))
    return group_img

def generate_subject_imgs(subject, output_dir, masker, output_type, file_id='test', overwrite=False):
    masker.standardize = False
    masker.detrend = False
    if not isinstance(output_type, list):
        output_type = [output_type]

    if isinstance(subject, list):
        for sub in subject:
            generate_subject_imgs(sub, output_dir, masker, output_type, file_id, overwrite)
        return

    output_type_filtered = []
    for ty in output_type:
        result_files = glob.glob(
            op.join(output_dir, '*'+subject+'*'+ty+'.nii.gz'))
        if len(result_files) > 0 :
            print('{} has existing {} results.'.format(subject, ty))
        else:
            output_type_filtered.append(ty)
    output_type = output_type_filtered
    if len(output_type) == 0:
        return
        
    score_files = glob.glob(
        op.join(output_dir, 'cache', '*'+subject+'*.npz'))
    if len(score_files) != 9:
        print('{} has no/corrupted score files.'.format(subject))
        return
    
    print('Regression | Generate {} imgs...'.format(subject))

    try:
        score_dump = np.load(score_files[0], mmap_mode='r')
        alpha_space = score_dump['alpha']
        dimension_space = score_dump['dimension']
        base_dump = score_dump['r2_test']
        if 'r2' in output_type:
            r2_test_score = np.zeros((len(score_files), *base_dump.shape), dtype=np.float64)
            r2_test_score[0, :] = base_dump
        if 'mse' in output_type:
            mse_test_score = 1 - base_dump
        if 'r' in output_type:
            r_test_score = np.sign(base_dump)*np.sqrt(np.abs(base_dump))

        for idx, score_file in enumerate(score_files[1:]):
            score_dump = np.load(score_file, mmap_mode='r')
            # assert subject == pickle.load(
            #     fi), '{} not aligned in file {}.'.format(subject, score_file)
            assert np.all(alpha_space == score_dump['alpha']), '{} has wrong alpha space in file {}.'.format(subject, score_file)
            assert np.all(dimension_space == score_dump['dimension']), '{} has wrong dim space in file {}.'.format(subject, score_file)
            base_dump = score_dump['r2_test']
            if 'r2' in output_type:
                r2_test_score[idx+1, :] = base_dump
            if 'mse' in output_type:
                mse_test_score += 1 - base_dump
            if 'r' in output_type:
                r_test_score += np.sign(base_dump)*np.sqrt(np.abs(base_dump))
    except AssertionError:
        print('{} has no/corrupted score files.'.format(subject))
        return
    
    if 'mse' in output_type:
        mse_test_score /= len(score_files)
    if 'r' in output_type:
        r_test_score /= len(score_files)

    if 'mse' in output_type:
        nib.save(masker.inverse_transform(mse_test_score.max(axis=(0, 1))), op.join(
            output_dir, 'test_{}_mse.nii.gz'.format(subject)))
    if 'r' in output_type:
        r_test_score = np.piecewise(r_test_score, [r_test_score <= -0.99, r_test_score >= 0.99], [0, 0, lambda x: x])
        nib.save(masker.inverse_transform(r_test_score.max(axis=(0, 1))), op.join(
            output_dir, 'test_{}_r.nii.gz'.format(subject)))

    test_score = np.piecewise(r2_test_score, [r2_test_score < 0, r2_test_score >= 0.99], [0, 0, lambda x: x])
    test_mean = test_score.mean(axis=0)
    test_var = test_score.var(axis=0)
    test_best_dim_id = test_mean.argmax(axis=0)
    test_best_dim_score = test_mean.max(axis=0)
    test_best_dim_best_alpha_id = test_best_dim_score.argmax(axis=0)
    test_best_dim_best_alpha_score = test_best_dim_score.max(axis=0)
    test_best_dim_id_of_best_alpha = test_best_dim_id[test_best_dim_best_alpha_id, range(
        test_mean.shape[-1])]
    test_best_dim_of_best_alpha = dimension_space[test_best_dim_id_of_best_alpha]
    test_best_dim_best_alpha = alpha_space[test_best_dim_best_alpha_id]
    test_best_dim_best_alpha_var = test_var[test_best_dim_id_of_best_alpha, test_best_dim_best_alpha_id, range(
        test_mean.shape[-1])]

    print('Saving to ', op.join(
        output_dir, 'test_{}_r2.nii.gz'.format(subject)))
    nib.save(masker.inverse_transform(test_best_dim_best_alpha_score), op.join(
        output_dir, 'test_{}_r2.nii.gz'.format(subject)))
    nib.save(masker.inverse_transform(test_best_dim_best_alpha_var), op.join(
        output_dir, 'test_{}_r2_var.nii.gz'.format(subject)))
    nib.save(masker.inverse_transform(test_best_dim_of_best_alpha), op.join(
        output_dir, 'test_{}_dim.nii.gz'.format(subject)))
    nib.save(masker.inverse_transform(test_best_dim_best_alpha), op.join(
        output_dir, 'test_{}_alpha.nii.gz'.format(subject)))
    return


def process_subject(subj_dir, subject, dtx_mat, output_dir, model_name, alpha_space, dimension_space, masker, core_number, verbose):
    if len(glob.glob(op.join(
            output_dir, 'test_{}_*.nii.gz'.format(subject)))) >= 1:
        print('Skip training {}, using cached file.'.format(subject), flush=True)
        return

    fmri_filenames = sorted(
        glob.glob(os.path.join(subj_dir, subject, "run*.nii.gz")))
    with warnings.catch_warnings(): 
        # Disable RuntimeWarning: invalid value encountered in sqrt std = np.sqrt((signals ** 2).sum(axis=0))
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fmri_runs = [masker.transform(f) for f in fmri_filenames]

    dim_alpha_search_with_log(fmri_runs, dtx_mat, alpha_space,
                              dimension_space, subject, model_name, output_dir, send_mail_log, core_number, verbose)
    gc.collect()
    # generate_subject_imgs(subject, output_dir, masker)


def main(dmtx_dir, subj_dir, output_dir, model_name, alpha_space, dimension_space, masker, output_type=['r2', 'mse', 'r'], core_number=-1, verbose=True):
    if not op.isdir(output_dir):
        os.mkdir(output_dir)

    cache_dir = op.join(output_dir, 'cache')
    if not op.isdir(cache_dir):
        os.mkdir(cache_dir)

    design_files = sorted(glob.glob(op.join(dmtx_dir, 'dmtx_?_ortho.csv')))
    if len(design_files) != 9:
        print("dmtx_?.csv files not found in %s" % dmtx_dir)
        sys.exit(1)
    dtx_mat0 = [pd.read_csv(df) for df in design_files]
    dtx_mat = [((dtx - dtx.mean()) / dtx.std()).values for dtx in dtx_mat0]

    if alpha_space is None:
        alpha_space = [0]

    if dimension_space is None:
        dimension_space = [dtx_mat[0].shape[1]]

    subjlist = [op.basename(f) for f in glob.glob(op.join(subj_dir, 'sub*'))]

    for idx, subject in enumerate(subjlist):
       
        msg = """Begin processing {}/{}: {} 
Searching space is:
    alpha : {}
    dim   : {}
""".format(idx, len(subjlist), subject, alpha_space, dimension_space)
        if verbose:
            print(msg, flush=True)
            send_mail_log('{} loop'.format(model_name), msg)
        process_subject(subj_dir, subject, dtx_mat, output_dir,
                        model_name, alpha_space, dimension_space, masker, core_number, verbose)
        generate_subject_imgs(subject, output_dir, masker, file_id='test', overwrite=False, output_type=output_type)
    return


if __name__ == '__main__':
    # parse command line
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "d:s:o:m:",
                                   ["design_matrices=",
                                    "subject_fmri_data=",
                                    "output_dir=", "model_name="])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for o, a in opts:
        if o in ('-m', '--model_name'):
            model_name = a
        elif o in ('-d', '--design_matrices'):
            dmtx_dir = a
        elif o in ('-s', '--subject_fmri_data'):
            subj_dir = a
        elif o in ('-o', '--output_dir'):
            output_dir = a

    with open(op.join(subj_dir, 'masker.pkl'), mode='rb') as fl:
        masker = pickle.load(fl)

    main(dmtx_dir, subj_dir, output_dir, model_name, None, None, masker)
