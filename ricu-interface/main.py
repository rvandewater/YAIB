import pathlib

# import rpy2
# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr
# print(rpy2.__version__)

r_files = ['.RData', '.RHistory', '.RProfile']
for rf in r_files:
    p = pathlib.Path.cwd() / rf
    try:
        p.rename(p.with_suffix('.ignore'))
    except FileNotFoundError:
        pass
import rpy2
from rpy2 import robjects
from rpy2.robjects.packages import importr
for rf in r_files:
    p = pathlib.Path.cwd() / rf
    p = p.with_suffix('.ignore')
    try:
        p.rename(p.with_suffix(''))
    except FileNotFoundError:
        pass


base = importr('base')
utils = importr('utils')
renv = importr('renv')
utils.chooseCRANmirror(ind=1)
renv.install('rtools')
# utils.install_packages('rtools')
# # utils.install_packages('renv')
#
renv.restore()
# ricu = importr('ricu')

# ricu.download_src("miiv", data_dir = "C:\\Users\\Robin van de Water\\Documents\\Datasets")