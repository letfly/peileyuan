import numpy as np
import scipy
import xgboost as xgb


# to load a libsvm text file or a XGBoost binary file into DMatrix
dtrain = xgb.DMatrix('train.svm.txt')
dtext = xgb.DMatrix('test.svm.buffer')

# to load a numpy array into DMatrix
data = np.random.rand(5, 10) # 5 entities, each contains 10 features
label = np.random.randint(2, size=5) # binary target
dtrain = xgb.DMatrix(data, label=label)

# to load a scipy.sparse array into DMatrix
csr = scipy.sparse.csr_matrix((dat, (row, col)))
dtrain = xgb.DMatrix(csr)
