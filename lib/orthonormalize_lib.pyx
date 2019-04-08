import sys
import getopt
import os
import glob
import os.path as op
import numpy as np
cimport numpy as np
import numpy.linalg as npl
# cimport numpy.linalg as npl
from numpy import (corrcoef, around, array, dot, identity, mean)
from numpy import column_stack as cbind
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def ortho_proj(np.ndarray Y, np.ndarray M):
    """ returns the orthogonal component of Y to the space spanned by M and the constant vector 1 """
    if M.ndim == 1:   # M is a vector but needs to be a 2-D matrix
        M = M[:, np.newaxis]
    cdef np.ndarray M2 = np.hstack((np.ones((len(M), 1), dtype=np.float), M))  # adding the constant 
    betas,_,_,_ = npl.lstsq(M2, Y)
    cdef np.ndarray Xc = np.dot(M2, betas)  # colinear component "residuals"
    return Y - Xc

def main_parallel(int r, str f, str output_dir):
    print("Run #%d:" % r)
    df = pd.read_csv(f)
    # M1 = df.values.T
    # print(around(corrcoef(M1), 2))

    # display = sns.pairplot(df)
    # fn, ext = op.splitext(op.basename(f))
    # display.savefig(op.join(output_dir, fn + '_nonortho.png'))
    df = (df - df.mean()) / df.std()
    # df.loc[:, df.columns[0]] = df.loc[:, df.columns[0]] - mean(df.loc[:, df.columns[0]])
    for idx, column in enumerate(df.columns[1:]):
        print("Run #%d: column %d" % (r, idx))
        df.loc[:, column] =  ortho_proj(df.loc[:, column].to_numpy(), df.loc[:, df.columns[:idx+1]].to_numpy())

    fname, ext  = op.splitext(op.basename(f))
    newfname = op.join(output_dir, fname + '_ortho' + ext)
    df.to_csv(newfname, index=False)
    # display = sns.pairplot(df)
    # display.savefig(op.join(output_dir, fn + '_ortho.png'))
    # M2 = df.values
    # print(around(corrcoef(M2.T), 2))
    plt.close('all')