import os
import sys

import sklearn.utils.estimator_checks

sys.path.append(os.pardir)
from SklearnWrapper import SklearnWrapperClassifier, SklearnWrapperRegressor

#estimator = SklearnWrapperClassifier(MLP(10, 10))
#sklearn.utils.estimator_checks.check_estimator(estimator)

print('Classifier test')
sklearn.utils.estimator_checks.check_estimator(SklearnWrapperClassifier)

print('Regressor test')
sklearn.utils.estimator_checks.check_estimator(SklearnWrapperRegressor)
