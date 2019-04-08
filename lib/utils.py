
import json
import os
import os.path as op
import numpy as np
import sys
import glob
import pickle
from . import generate_regressors, merge_regressors, orthonormalize, check_design_matrices, regressions
# import .generate_regressors, .merge_regressors, .orthonormalize, .check_design_matrices
# import regressions
from nilearn.input_data import MultiNiftiMasker
from nilearn.masking import apply_mask, compute_multi_epi_mask



class Env:
    def __init__(self, root_dir, lingua):
        self.root_dir = op.abspath(root_dir)
        self.lingua = lingua
        self.onset_dir = op.join(self.root_dir, 'data', 'onsets', lingua)
        self.reg_dir = op.join(self.root_dir, 'data', 'regressors', lingua)
        self.fmri_dir = op.join(self.root_dir, 'data', 'fmri', lingua)
        self.model_base_dir = op.join(self.root_dir, 'models', lingua)
        self.model_comp_dir = op.join(self.root_dir, 'cross_models', lingua)
        self.verbose = False
        self.subjlist = None
        for diry in [self.onset_dir, self.reg_dir, self.fmri_dir, self.model_base_dir, self.model_comp_dir]:
            if not op.isdir(diry):
                os.makedirs(diry)
    
    def get_available_subject(self):
        if self.subjlist:
            return self.subjlist
        self.subjlist = [op.basename(f) for f in glob.glob(op.join(self.fmri_dir, 'sub*'))]
        return self.subjlist
    
    def get_masker(self):
        file_path = op.join(self.fmri_dir, 'masker.pkl')
        if op.isfile(file_path):
            with open(file_path, mode='rb') as fl:
                masker = pickle.load(fl)
        else:
            imgs = glob.glob(os.path.join(self.fmri_dir, 'sub*', "run*.nii.gz"))
            if len(imgs) > 0:
                global_mask = compute_multi_epi_mask(imgs, n_jobs=-1)
                # masks = [compute_epi_mask(glob.glob(os.path.join(rootdir, s, "run*.nii.gz"))) for s in subjects]
                # global_mask = math_img('img>0.5', img=mean_img(masks))
                masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True)
                masker.fit()
                with open(file_path, mode='wb') as fl:
                    pickle.dump(masker, fl, protocol=-1)
            else:
                print('No fmri data found', os.path.join(
                    self.fmri_dir, 'sub_*', "run*.nii.gz"))
                exit(0)
        return masker


class Model:
    def __init__(self, model_name, environment):
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

    def config_model(self, config_file: str=None, overwrite=False):
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
            embedding_regs = [config['embedding']['name_base']+str(idx) for idx in range(1, config['embedding']['dim']+1)]

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
    """.format(self.model_name, self.environment.lingua, \
    len(self.alpha), ', '.join(map(str,self.alpha)), len(self.dimension), \
        ', '.join(map(str, self.dimension)), len(self.regressors), ', '.join(self.regressors)))
        else:
            print('Model not configured.')
        return

    def generate_regressors(self):
        generate_regressors.main(self.environment.lingua, \
            self.environment.reg_dir, self.environment.onset_dir, False, self.regressors, None, None)
        return

    def generate_design_matrices(self):
        merge_regressors.main(self.regressors, self.environment.reg_dir, \
            self.design_matrix_dir, False)
        if self.orthonormalize:
            orthonormalize.main(self.design_matrix_dir, self.design_matrix_dir, False)
        if self.environment.verbose:
            files = glob.glob(op.join(self.design_matrix_dir, '*.csv'))
            check_design_matrices.main(files)
        return

    def generate_individual_results(self):
        regressions.main(self.design_matrix_dir, self.environment.fmri_dir, \
            self.first_level_results, self.model_name, self.alpha, self.dimension, self.environment.get_masker())
        return 

    def generate_group_results(self):
        regressions.generate_group_imgs(self.environment.get_available_subject(), 
        self.first_level_results, self.group_level_results)
        return



# class ModelComparison:
#     def __init__(self, model_name, environment):