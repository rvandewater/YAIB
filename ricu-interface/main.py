import pathlib

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
print(rpy2.__version__)
def main():

    # WORKAROUND FOR RPY2 IMPORT ERROR
    # r_files = ['.RData', '.RHistory', '.RProfile']
    # for rf in r_files:
    #     p = pathlib.Path.cwd() / rf
    #     try:
    #         p.rename(p.with_suffix('.ignore'))
    #     except FileNotFoundError:
    #         pass
    # import rpy2
    # from rpy2 import robjects
    # from rpy2.robjects.packages import importr
    #
    # for rf in r_files:
    #     p = pathlib.Path.cwd() / rf
    #     p = p.with_suffix('.ignore')
    #     try:
    #         p.rename(p.with_suffix(''))
    #     except FileNotFoundError:
    #         pass


    base = importr('base')
    utils = importr('utils')
    ricu = importr('ricu')

    demo = ricu.load_src("admissions", "mimic_demo")
    print(demo)
    concepts = ricu.load_concepts("hr", "mimic_demo", verbose = True)
    print(concepts)

    r_source = robjects.r['source']

    # Load mortality source file
    r_source("mortality.R")


def install_requirements():
    base = importr('base')
    utils = importr('utils')
    utils.chooseCRANmirror(ind=1)
    utils.install_packages('rlang')
    utils.install_packages("mimic.demo", repos="https://eth-mds.github.io/physionet-demo")
    utils.install_packages("ricu")
    utils.install_packages("argparser")
    utils.install_packages("arrow")



if __name__ == "__main__":
    main()