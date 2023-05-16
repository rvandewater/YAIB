import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
print(rpy2.__version__)

base = importr('base')
utils = importr('utils')
utils.chooseCRANmirror(ind=1)
utils.install_packages('renv')
renv = importr('renv')

renv.record("renv@0.17.3")
renv.restore()
ricu = importr('ricu')

# ricu.download_src("miiv", data_dir = "C:\\Users\\Robin van de Water\\Documents\\Datasets")