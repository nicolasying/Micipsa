
import glob
import inspect
import json
import os
import os.path as op
import pickle
import sys

import matplotlib.pyplot as plt
from matplotlib import cm
import nibabel as nib
import numpy as np
from nilearn.image import math_img, mean_img
from nilearn.plotting import plot_glass_brain
# import .generate_regressors, .merge_regressors, .orthonormalize, .check_design_matrices
# import regressions
from nilearn.input_data import MultiNiftiMasker
from nilearn.masking import apply_mask, compute_multi_epi_mask

from . import (check_design_matrices, generate_regressors, merge_regressors,
               orthonormalize, regressions)


def display_for_subject(dic):
    f, ax = plt.subplots((len(dic)+1)//2, 2, figsize=(30, 3*len(dic)))
    for idx, (key, val) in enumerate(dic.items()):
        plot_glass_brain(val, display_mode='lzry', colorbar=True, title=key, plot_abs=False,
                         figure=f, axes=ax[idx % ((len(dic)+1)//2), idx//((len(dic)+1)//2)], cmap=cm.coolwarm)
    return f


class Env:
    def __init__(self, root_dir, lingua):
        self.root_dir = op.abspath(root_dir)
        self.lingua = lingua
        self.onset_dir = op.join(self.root_dir, 'data', 'onsets', lingua)
        self.reg_dir = op.join(self.root_dir, 'data', 'regressors', lingua)
        self.fmri_dir = op.join(self.root_dir, 'data', 'fmri', lingua)
        self.model_base_dir = op.join(self.root_dir, 'models', lingua)
        self.model_comp_dir = op.join(self.root_dir, 'model_contrasts', lingua)
        self.verbose = False
        self.subjlist = None
        for diry in [self.onset_dir, self.reg_dir, self.fmri_dir, self.model_base_dir, self.model_comp_dir]:
            if not op.isdir(diry):
                os.makedirs(diry)

    def get_available_subject(self):
        if self.subjlist:
            return self.subjlist
        self.subjlist = [op.basename(f) for f in glob.glob(
            op.join(self.fmri_dir, 'sub*'))]
        return self.subjlist

    def get_masker(self):
        file_path = op.join(self.fmri_dir, 'masker.pkl')
        if op.isfile(file_path):
            with open(file_path, mode='rb') as fl:
                masker = pickle.load(fl)

        else:
            imgs = glob.glob(os.path.join(
                self.fmri_dir, 'sub*', "run*.nii.gz"))
            if len(imgs) > 0:
                global_mask = compute_multi_epi_mask(imgs, n_jobs=-1)
                # masks = [compute_epi_mask(glob.glob(os.path.join(rootdir, s, "run*.nii.gz"))) for s in subjects]
                # global_mask = math_img('img>0.5', img=mean_img(masks))
                masker = MultiNiftiMasker(
                    global_mask, detrend=True, standardize=True)
                masker.fit()
                with open(file_path, mode='wb') as fl:
                    pickle.dump(masker, fl, protocol=-1)
            else:
                print('No fmri data found', os.path.join(
                    self.fmri_dir, 'sub_*', "run*.nii.gz"))
                exit(0)
        return masker


class Model:
    def __init__(self, model_name, environment: Env):
        self.model_name = model_name
        self.environment = environment
        self.model_dir = op.join(environment.model_base_dir, model_name)
        self.design_matrix_dir = op.join(self.model_dir, 'design_matrices')
        self.first_level_results = op.join(
            self.model_dir, 'first_level_results')
        self.group_level_results = op.join(
            self.model_dir, 'group_level_results')
        self.configured = False
        for diry in [self.model_dir, self.design_matrix_dir, self.first_level_results, self.group_level_results]:
            if not op.isdir(diry):
                os.makedirs(diry)

    def config_model(self, config_file: str = None, overwrite=False):
        # Read Model Configuration Files
        if not self.configured or overwrite:
            print("Model: Configuring model from", config_file)
        else:
            print("Model: Model is already configured")
            return

        if config_file is None:
            config_file = op.join(self.model_dir, 'config.json')
            print("Model: using default config file path.")
        with open(config_file, mode='r') as fi:
            config = json.load(fp=fi)

        if 'lingua' in config and config['lingua'] != self.environment.lingua:
            print("Model Config: lingua overwrite from {} to {}.".format(
                self.environment.lingua, config['lingua']))
            self.environment.lingua = config['lingua']

        if 'model_name' in config and config['model_name'] != self.model_name:
            print("Model Config: model_name overwrite from {} to {}.".format(
                self.model_name, config['model_name']))
            self.model_name = config['model_name']

        if 'embedding' not in config or 'dim' not in config['embedding'] \
            or config['embedding']['dim'] == 0 or 'name_base' not in config['embedding'] \
                or len(config['embedding']['name_base']) == 0:
            embedding_regs = []
        elif isinstance(config['embedding']['regressors'], list):
            embedding_regs = config['embedding']['regressors']
        else:
            embedding_regs = [config['embedding']['name_base'] +
                              str(idx) for idx in range(1, config['embedding']['dim']+1)]

        if 'base_regressors' not in config:
            base_regressors = []
        elif isinstance(config['base_regressors'], list):
            base_regressors = config['base_regressors']
        else:
            print("Model Config: base_regressors format not recongized.",
                  config['base_regressors'])
            base_regressors = []

        self.regressors = base_regressors + embedding_regs

        if 'alpha' not in config:
            alpha = [0]
        elif isinstance(config['alpha'], list):
            alpha = []
            for ele in config['alpha']:
                if isinstance(ele, dict):
                    alpha += list(np.logspace(
                        ele['start'], ele['end'], ele['step']))
                elif isinstance(ele, int) or isinstance(ele, float):
                    alpha.append(ele)
                else:
                    print("Model Config: alpha format not recongized.. skip..", ele)
        else:
            print("Model Config: alpha format not recongized.",
                  config['alpha'])
            alpha = [0]
        self.alpha = alpha

        if 'dimension' in config and isinstance(config['dimension'], list):
            dimension = []
            for ele in config['dimension']:
                if isinstance(ele, dict):
                    dimension += list(range(
                        ele['start'], ele['end'], ele['step']))
                elif isinstance(ele, int):
                    dimension.append(ele)
                else:
                    print("Model Config: dimension format not recongized.. skip..", ele)
        else:
            print("Model Config: dimension format not recongized.",
                  config['dimension'])
            dimension = [len(self.regressors)]
        self.dimension = dimension

        if 'orthonormalize' in config:
            self.orthonormalize = config['orthonormalize']
        else:
            self.orthonormalize = True
        self.configured = True

    def print_model_config(self, out=sys.stdout):
        if self.configured:
            out.write("""Model Config:
            ==================================
            Model : {:>30}    | Language  : {:>5}
            Alpha search space ({:4d})    : {}
            Dimension search space ({:4d}): {}
            Regressors ({:4d})            :  {}
            ==================================
                """.format(self.model_name, self.environment.lingua,
                           len(self.alpha), ', '.join(
                               map(str, self.alpha)), len(self.dimension),
                           ', '.join(map(str, self.dimension)), len(self.regressors), ', '.join(self.regressors)))
        else:
            print('Model not configured.')
        return

    def generate_regressors(self):
        generate_regressors.main(self.environment.lingua,
                                 self.environment.reg_dir, self.environment.onset_dir, False, self.regressors, None, None)
        return

    def generate_design_matrices(self):
        merge_regressors.main(self.regressors, self.environment.reg_dir,
                              self.design_matrix_dir, False)
        if self.orthonormalize:
            orthonormalize.main(self.design_matrix_dir,
                                self.design_matrix_dir, False)
        if self.environment.verbose:
            files = glob.glob(op.join(self.design_matrix_dir, '*.csv'))
            check_design_matrices.main(files)
        return

    def generate_individual_results(self, core_number=-1, verbose=None, output_type=['r2', 'mse', 'r']):
        if verbose is None:
            verbose = self.environment.verbose
        regressions.main(dmtx_dir=self.design_matrix_dir, subj_dir=self.environment.fmri_dir,
                         output_dir=self.first_level_results, model_name=self.model_name, alpha_space=self.alpha, dimension_space=self.dimension,
                         masker=self.environment.get_masker(), core_number=core_number, verbose=verbose, output_type=output_type)
        return

    def check_individual_results(self, generate_if_missing=False):
        res_files = glob.glob(
            op.join(self.first_level_results, 'test_*_r2.nii.gz'))
        if len(self.environment.get_available_subject()) == len(res_files):
            return True
        elif generate_if_missing:
            self.generate_individual_results()
            return self.check_individual_results(False)
        else:
            print("Model: Got {} first level results for {} subjects.".format(
                len(res_files), len(self.environment.get_available_subject())))
            return False

    def generate_group_results(self, output_type=['r2', 'mse', 'r']):
        regressions.generate_group_imgs(self.environment.get_available_subject(),
                                        self.first_level_results, self.group_level_results, file_id='test', result_type=output_type)
        return

    def get_result_for_subject(self, subject):
        res = glob.glob(op.join(self.first_level_results,
                                'test_{}_r2.nii.gz'.format(subject)))
        if len(res) != 1:
            print("Model: Check manually results found for", subject)
            return
        return res[0]


class ModelComparison:
    def __init__(self, comparison_name, environment: Env):
        self.comparison_name = comparison_name
        self.environment = environment
        self.comp_dir = environment.model_comp_dir
        self.first_level_results = op.join(
            self.comp_dir, 'first_level_results')
        self.group_level_results = op.join(
            self.comp_dir, 'group_level_results')
        self.report_dir = op.join(self.comp_dir, self.comparison_name)
        self.configured = False
        for diry in [self.comp_dir, self.first_level_results, self.group_level_results, self.report_dir]:
            if not op.isdir(diry):
                os.makedirs(diry)

    def config_comp(self, config_file: str = None, overwrite=False):
        # Read ModelComp Configuration Files
        if not self.configured or overwrite:
            print("ModelComparison: Configuring comparison from", config_file)
        else:
            print("ModelComparison: Comparison is already configured")
            return

        if config_file is None:
            config_file = op.join(
                self.comp_dir, '{}.json'.format(self.comparison_name))
            print("ModelComparison: using default config file path.")
        with open(config_file, mode='r') as fi:
            config = json.load(fp=fi)

        try:
            if 'lingua' in config:
                assert config['lingua'] == self.environment.lingua, "ModelComparison: lingua config ({}) mismatch with environment setting ({}).".format(
                    config['lingua'], self.environment.lingua)
            if 'comparison_name' in config and config['comparison_name'] != self.comparison_name:
                print("ModelComparison: Overwriting comparison name from {} to {}.".format(
                    self.comparison_name, config['comparison_name']))

            if 'models' in config and len(config['models']) > 0:
                self.models = {model_name: Model(
                    model_name, self.environment) for model_name in config['models']}
                for model in self.models:
                    self.models[model].config_model()
            else:
                print("ModelComparison: No base model found in config.")
                return

            if 'contrast' in config and len(config['contrast']) > 0:
                self.contrast = config['contrast']
            else:
                print("ModelComparison: No model contrast found in config.")
                return
        except:
            return

        self.configured = True
        return

    def print_comp_config(self, out=sys.stdout):
        if self.configured:
            contrast_strings = ["-{:>2d}| {:>10}: {:>10} - {:>10}".format(
                idx, comp_name, models[0], models[1]) for idx, (comp_name, models) in enumerate(self.contrast.items())]
            out.write("""ModelComparison Config:
                ==================================
                Comparison Name : {:>30}    | Language  : {:>5}
                Base Models ({:>4d}): {}
                Contrasts   ({:>4d}):
                {}
                ==================================
                """.format(self.comparison_name, self.environment.lingua, len(self.models), ', '.join(self.models.keys()), len(self.contrast), '\n'.join(contrast_strings)))
        else:
            print('ModelComparison not configured.')
        return

    def generate_individual_results(self, contrast=None):
        if not self.configured:
            self.config_comp()
        if contrast is None:
            contrast = list(self.contrast.keys())
        if isinstance(contrast, list):
            for con in contrast:
                self.generate_individual_results(con)
        elif isinstance(contrast, str) and contrast in self.contrast:
            models_to_compare = self.contrast[contrast]
            base_model = self.models[models_to_compare[1]]
            aug_model = self.models[models_to_compare[0]]

            output_dir = op.join(self.first_level_results,
                                 models_to_compare[0]+'_'+models_to_compare[1])

            if not op.isdir(output_dir):
                os.makedirs(output_dir)
            if base_model.check_individual_results(generate_if_missing=False) and aug_model.check_individual_results(generate_if_missing=False):
                for subject in self.environment.get_available_subject():
                    base_model_img = base_model.get_result_for_subject(subject)
                    aug_model_img = aug_model.get_result_for_subject(subject)
                    self._contrast_image(aug_model_img, base_model_img, op.join(
                        output_dir, 'test_{}_r2.nii.gz'.format(subject)))
            else:
                print("ModelComp: Missing regression results for compared models ({} / {})".format(
                    base_model.model_name, aug_model.model_name))

        else:
            print("ModelComp: Unrecognized contrast name.\nBe sure to input contrast among {}".format(
                ", ".join(self.contrast.keys())))
        return

    def generate_group_results(self, contrast=None):
        if contrast is None:
            contrast = list(self.contrast.keys())
        if isinstance(contrast, list):
            for con in contrast:
                self.generate_group_results(con)
        elif isinstance(contrast, str) and contrast in self.contrast:
            models_to_compare = self.contrast[contrast]
            output_dir = op.join(self.group_level_results,
                                 models_to_compare[0]+'_'+models_to_compare[1])
            input_dir = op.join(self.first_level_results,
                                models_to_compare[0]+'_'+models_to_compare[1])
            regressions.generate_group_imgs(self.environment.get_available_subject(),
                                            input_dir, output_dir)
        return

    def generate_report(self, subject=None, result_type='r2', file_id='test'):
        if not self.configured:
            self.config_comp()

        # Generate Individual
        if subject is None:
            for subject in self.environment.get_available_subject():
                self.generate_report(
                    subject=subject, result_type=result_type, file_id=file_id)
            self.generate_report(
                subject='group', result_type=result_type, file_id=file_id)
            return

        if isinstance(result_type, list):
            for ty in result_type:
                self.generate_report(
                    subject=subject, result_type=ty, file_id=file_id)
            return

        if subject == 'group':
            dir_selector = 'group_level_results'
        else:
            dir_selector = 'first_level_results'

        plot_dict = dict()
        for model_name, model in self.models.items():
            plot_dict[model_name.replace('-', '+')] = op.join(
                model.model_dir, dir_selector, '{}_{}_{}.nii.gz'.format(file_id, subject, result_type))
        for constrast, models_to_compare in self.contrast.items():
            plot_dict[constrast] = op.join(self.comp_dir, dir_selector, models_to_compare[0] +
                                           '_'+models_to_compare[1], '{}_{}_{}.nii.gz'.format(file_id, subject, result_type))
        print("ModelComp: Printing Report for {} {}".format(subject, file_id))
        f = display_for_subject(plot_dict)
        f.suptitle(subject)
        plt.savefig(op.join(self.report_dir, '{}_{}_{}.png'.format(
            file_id, subject, result_type)))
        plt.close('all')

    @staticmethod
    def _contrast_image(im1, im2, file_path):
        if file_path is not None:
            if op.isfile(file_path):
                return nib.load(file_path)

        if isinstance(im1, str):
            im1 = nib.load(im1)
        if isinstance(im2, str):
            im2 = nib.load(im2)
        contrast = math_img("im1 - im2", im1=im1, im2=im2)
        if file_path is not None:
            nib.save(contrast, file_path)
        return contrast
