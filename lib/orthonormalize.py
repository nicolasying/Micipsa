#! /usr//bin/env python
# Time-stamp: <2017-07-19 22:44:53 cp983411>


""" read the design matrices dmt_*.csv and perform a sequential orthogonalization of the variables """

import sys
import getopt
import os
import glob
import os.path as op
import numpy as np
import numpy.linalg as npl
from numpy import (corrcoef, around, array, dot, identity, mean)
from numpy import column_stack as cbind
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from orthonormalize_lib import main_parallel

def main(data_dir, output_dir, overwrite=False):
    filter = op.join(data_dir, 'dmtx_?.csv')
    dfiles = glob.glob(filter)
    if len(dfiles) == 0:
        print("Cannot find files "+ filter)
        return
    Parallel(n_jobs=-2)(delayed(main_parallel)(r, f, output_dir) for r, f in enumerate(dfiles))
    return


if __name__ == '__main__':
    data_dir = '.'
    output_dir = '.'
    
    # parse command line to change default
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "d:o:w:",
                                   ["design_matrices=", "output_dir=","--no-overwrite"])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    
    overwrite = True
    for o, a in opts:
        if o in ('-d', '--design_matrices'):
            data_dir = a
        elif o in ('-o', '--output_dir'):
            output_dir = a
        elif o == '--no-overwrite':
            overwrite = False
    main(data_dir, output_dir, overwrite)