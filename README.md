Scripts for the analysis of the fMRI data of "The Little Prince" project (Hale, Pallier)
Micipsa - Songsheng YING
----------------------------------------------------------------------------------------
soshyng@gmail.com

Modified from [Christophe Pallier](mailto:Christophe@pallier.org)'s [git](https://github.com/chrplr/lpp-scripts3)
which allows to perform fMRI encoding experiments with regressions.

Requirements:

- Python: pandas, nistats, nibabel, nilearn, statsmodels

# Differences from the original git
1. Pythonize the original `makefile` and use class methods to build regressors from onsets, design matrices from regressors.
2. Use config files to define model parameters and model comparison settings.
3. File organization by model and modelComp.
4. Use Ridge regression as the default regression model instead of GLM.
5. Generates r2, MSE scores for whole models, no longer provides feature-wise contrast maps.
6. Large GridSearch with shrinkage: step-wise forward feature selection & alpha.

