#! /usr/bin/env python
#! Time-stamp: <2018-04-08 10:06:29 cp983411>

""" take a list of predictor names on the command line and compute the hrf convolved regressors from the corresponding onset files [1-0]_name.csv """

import argparse
import os
import os.path as op
import sys
import warnings

from joblib import Parallel, delayed

from .events2reg import process_onefile

warnings.filterwarnings("ignore", category=DeprecationWarning) 

def main(lingua, output_dir, input_dir, overwrite, regressors, nscans, nblocks):
    if lingua == 'en' and (not nscans or len(nscans) == 0):
        nscans = [282, 298, 340, 303, 265, 343, 325, 292, 368]  # numbers of scans in each session
    elif lingua == 'fr' and (not nscans or len(nscans) == 0):
        nscans = [309, 326, 354, 315, 293, 378, 332, 294, 336]  # numbers of scans in each session

    parameters = [('%d_%s.csv' % (1 + session, reg), ns)
                for reg in regressors for (session, ns) in enumerate(nscans)]

    Parallel(n_jobs=-2)(delayed(process_onefile) \
                        (op.join(input_dir, filen), 2.0, ns, overwrite, output_dir) for (filen, ns) in parameters)
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate fMRI regressors from [1-9]_REG.csv onsets files)")
    parser.add_argument("--output-dir", type=str, default='.')
    parser.add_argument("--input-dir", type=str, default='.')
    parser.add_argument('--no-overwrite', dest='overwrite', action='store_false')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true')
    parser.set_defaults(overwrite=False)
    parser.add_argument('regressors',
                        nargs='+',
                        action="append",
                        default=[])
    parser.add_argument('--nscans', nargs='+', action="append", default=[])
    parser.add_argument('--blocks', nargs='+', action="append", default=[])
    parser.add_argument('--lingua', type=str)

    args = parser.parse_args()
    assert len(args.nscans) == len(args.blocks), 'number of scans is not equal to number of blocks'
    regressors = args.regressors[0]

    main(args.lingua, args.output_dir, args.input_dir, args.overwrite, regressors, args.nscans, args.nblocks)
