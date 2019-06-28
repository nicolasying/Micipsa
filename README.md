# Scripts for the analysis of the fMRI data of "The Little Prince" project (Hale, Pallier)

Micipsa - Songsheng YING

----------------------------------------------------------------------------------------

soshyng@gmail.com

Modified from [Christophe Pallier](mailto:Christophe@pallier.org)'s [git](https://github.com/chrplr/lpp-scripts3)
which allows to perform fMRI encoding experiments with regressions.

Requirements:

- Python: pandas, nistats, nibabel, nilearn, statsmodels

## Differences from the original git

1. Pythonize the original `makefile` and use class methods to build regressors from onsets, design matrices from regressors.
2. Use config files to define model parameters and model comparison settings.
3. File organization by model and modelComp.
4. Use Ridge regression as the default regression model instead of GLM.
5. Generates r2, MSE scores for whole models, no longer provides feature-wise contrast maps.
6. Large GridSearch with shrinkage: step-wise forward feature selection & alpha.

## How to make it work 

### General guidelines

Three class interfaces are created in ```lib/utils``` to manage file paths (```Env```), model configuration and training (```Model```) and cross-model comparison (```ModelComparison```). In any case of non-clarity of the following text, please refer to the raw code.

### Initiating the environment 

1. Clone the original repo

2. Execute the following code from the root directory to create necessary folders

```
from lib import utils
env = utils.Env('./', <lingua>)
``` 

3. Load event onsets (generation of semantic onsets in [another repo](https://github.com/nicolasying/Micipsa-Text-Preprocessing)) into ```./data/onsets/<lingua>```, ```<lingua>``` in the Micipsa project should be set to ```fr```.

4. Load normalized fMRI data file in ```./data/fmri/<lingua>/<subject>/<run.nii.gz>```

### Creating regression models 

1. Create model configurations in ```./models/<lingua>/<model_name>``` by creating ```config.json```. Examples are available in the existing repo. 

2. In the configuration file, ```base_regressors``` will ask the model to look in onset folders with exact-name-matching onset files. E.g. ```rms``` will point to ```<run_id>_rms.csv```

3. ```embedding``` files will sequentially point to a series of onset files. It contains three subfields,  with ```dim``` an integer value, ```name_base``` serving as onset file-name filters, and ```regressors``` which takes either a list of onset feature names, or a string ```infer```, which would look for onset file starting with ```name_base``` and ending with an integer ranging from ```0``` to ```dim```. E.g. in ```rms-wrate-cwrate-sim100``` model, a list of regressor names are passed. In ```rms-wrate-cwrate-asn200```, ```<run_id>_<name_base><dim_range>.csv``` are consulted. 

4. ```alpha``` takes a list, of which members can either be a number, or a dictionary composed of three mandatory keys: ```start```, ```end``` and ```step```, which would be inflated into a log range. 

5. ```dimension``` is similar to ```alpha```, but the ranges are linear. 

6. ```orthonormalize``` indicates if regressor orthonormalization should be performed at the creation of design matrices. If this field is absent, the model would set the value to true by default.


### Model manipulation pipeline

1. Create model python object

```
model = utils.Model('<model_name>', env)
```

2. Configure model with configuration file, or by default with ```config.json``` in the corresponding folder

```
model.config_model()
```

3. Convolute event onsets to obtain separate regressors (regressors are shared across models)

```
model.generate_regressors()
```

4. Merge regressors into design matrices in the model local folder, check `model.orthonormalize` and `env.verbose`. Verbose flag would print and generate design matrix analysis files.

```
model.generate_design_matrices()
```

5. Finally the regression! The code will perform 9 cross validations (as the recordings are divided into 9 blocks both in English and French experiment), leaving 1 run out as validation data and 8 as training data. Voxel models are considered as independent and a uniform brain masker will be applied to transform the 3d fMRI array into an 1d array. The regression will store run-wise validation scores with each combination of ```alpha``` and ```dimension```. 

```
model.generate_individual_results(core_number=-1, verbose=None, output_type=['r2', 'mse', 'r'])
```

6. Group averages can be calculated with the following code. 

```
model.generate_group_results()
```

### Model comparison pipeline

The results generated with `ModelComparison` class is not used in the master's thesis. 

Model comparison configuration files are located at `./model_contrasts/<lingua>/<comparison_name>`.