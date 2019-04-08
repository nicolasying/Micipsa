#! /usr/bin/env python
# Time-stamp: <2017-07-20 17:10:17 cp983411>

""" read a series of regressor files, containing each 1 numeric column
and paste them horizontally to create 9 matrices in csv file format.
On the comand line list the name of the vectors; 
The filenames must have the pattern '[1-9]_%s_reg.csv' where %S is the name of the vector """

import sys
import os
import getopt
import os.path as op
import pandas as pd


def main(regs, input_dir, output_dir, overwrite=False, nruns=9):
    # merge columns into matrices
    for run in range(1, nruns + 1):
        fn = op.join(output_dir, "dmtx_{}.csv".format(run))
        if op.isfile(fn) and not overwrite:
            print('Using existing design matrix', fn)
            continue
        a = []
        for s in regs:
            fn = op.join(input_dir, "%s_%s_reg.csv" % (run, s))
            a.append(pd.read_csv(fn, header=None))
        merge = pd.concat(a, axis=1)
        merge.columns = regs
        fn = op.join(output_dir, "dmtx_{}.csv".format(run))
        merge.to_csv(fn, index=False)
    return


if __name__ == "__main__":
    # parse command line
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                "i:o:",
                                ["input_dir=", "output_dir=", "no-overwrite"])

    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    overwrite = True
    for o, a in opts:
        if o in ('-i', '--input_dir'):
            input_dir = a
        elif o in ('-o', '--output_dir'):
            output_dir = a
        elif o == "--no-overwrite":
            overwrite = False
                            
    if not op.isdir(output_dir):
        os.mkdir(output_dir)
    
    main(args, input_dir, output_dir, overwrite, nruns=9)
