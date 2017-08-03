import unittest

import chainer
import numpy
from chainer import links, testing, functions, optimizers
from chainer.datasets import TupleDataset
from chainer.testing import attr
from mock import mock
import matplotlib
from sklearn.datasets import load_iris

matplotlib.use('Agg')
import matplotlib.pyplot as plt


from chainer_sklearn.links import SklearnWrapperClassifier

class MLP(chainer.Chain):
    def __init__(self):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = links.Linear(None, 30)
            self.l2 = links.Linear(None, 3)

    def __call__(self, x):
        h = functions.relu(self.l1(x))
        h = self.l2(h)
        return h


class TestSklearnWrapperClassifier(unittest.TestCase):

    def setUp(self):
        # Load the iris dataset
        data, target = load_iris(return_X_y=True)
        self.train_x = data
        self.train_y = target
        self.label_dim = numpy.max(target) + 1

        X = data.astype(numpy.float32)
        y = target.astype(numpy.int32)
        self.dataset = TupleDataset(X, y)

        #self.train_x = numpy.arange(7).astype(numpy.float32)[:, None] / 1.
        #self.train_y = numpy.asarray([0, 0, 0, 0, 0, 1, 1]).astype(numpy.int32)
        #self.dataset = TupleDataset(self.train_x.copy(), self.train_y.copy())

    def test_array(self):
        #self.clf = SklearnWrapperClassifier(links.Linear(None, 4))
        self.clf = SklearnWrapperClassifier(MLP())
        self.check_fit_array()
        # These should be called after `fit`
        self.check_predict_array()
        self.check_predict_proba_array()
        self.check_score_array()

    def test_dataset(self):
        self.clf = SklearnWrapperClassifier(MLP())
        self.check_fit_dataset()
        # These should be called after `fit`
        self.check_predict_dataset()
        self.check_predict_proba_dataset()
        self.check_score_dataset()

    def check_fit_array(self):
        clf = self.clf.fit(self.train_x, self.train_y,
                           optimizers=optimizers.Adam())
        self.assertIsInstance(clf, SklearnWrapperClassifier)

    def check_fit_dataset(self):
        clf = self.clf.fit(self.dataset)
        self.assertIsInstance(clf, SklearnWrapperClassifier)

    def check_predict_array(self):
        y = self.clf.predict(self.train_x)
        num_correct = numpy.sum(y == self.train_y)
        self.assertGreaterEqual(num_correct / len(self.train_y), 0.5)

    def check_predict_dataset(self):
        y = self.clf.predict(self.dataset, retain_inputs=True)
        x_in, y_in = self.clf.inputs
        num_correct = numpy.sum(y == y_in)
        self.assertGreaterEqual(num_correct / len(y_in), 0.5)

    def check_predict_proba_array(self):
        y = self.clf.predict_proba(self.train_x)
        self.assertTrue(y.dtype == numpy.float32)
        self.assertTrue(y.shape == (len(self.train_x), self.label_dim))

    def check_predict_proba_dataset(self):
        pass

    def check_score_array(self):
        score = self.clf.score(self.train_x, self.train_y)
        self.assertGreaterEqual(score, 0.5)

    def check_score_dataset(self):
        pass

    def check_grid_search_array(self):
        pass

    def check_grid_search_dataset(self):
        pass

    def check_randomized_search_array(self):
        pass

    def check_randomized_search_dataset(self):
        pass


# testing.parameterize takes a list of dictionaries.
# Currently, we cannot set a function to the value of the dictionaries.
# As a workaround, we wrap the function and invoke it in __call__ method.
# See issue #1337 for detail.
class AccuracyWithIgnoreLabel(object):
    def __call__(self, y, t):
        return functions.accuracy(y, t, ignore_label=1)


@testing.parameterize(*testing.product({
    'accfun': [AccuracyWithIgnoreLabel(), None],
    'compute_accuracy': [True, False],
    'x_num': [1, 2]
}))
class TestClassifier(unittest.TestCase):

    def setUp(self):
        if self.accfun is None:
            self.link = SklearnWrapperClassifier(chainer.Link())
        else:
            self.link = SklearnWrapperClassifier(chainer.Link(),
                                         accfun=self.accfun)
        self.link.compute_accuracy = self.compute_accuracy

        self.x = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)
        self.t = numpy.random.randint(3, size=5).astype(numpy.int32)

    def check_call(self):
        xp = self.link.xp

        y = chainer.Variable(xp.random.uniform(
            -1, 1, (5, 7)).astype(numpy.float32))
        self.link.predictor = mock.MagicMock(return_value=y)

        x = chainer.Variable(xp.asarray(self.x))
        t = chainer.Variable(xp.asarray(self.t))
        if self.x_num == 1:
            loss = self.link(x, t)
            self.link.predictor.assert_called_with(x)
        elif self.x_num == 2:
            x_ = chainer.Variable(xp.asarray(self.x.copy()))
            loss = self.link(x, x_, t)
            self.link.predictor.assert_called_with(x, x_)

        self.assertTrue(hasattr(self.link, 'y'))
        self.assertIsNotNone(self.link.y)

        self.assertTrue(hasattr(self.link, 'loss'))
        xp.testing.assert_allclose(self.link.loss.data, loss.data)

        self.assertTrue(hasattr(self.link, 'accuracy'))
        if self.compute_accuracy:
            self.assertIsNotNone(self.link.accuracy)
        else:
            self.assertIsNone(self.link.accuracy)

    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()


class TestInvalidArgument(unittest.TestCase):

    def setUp(self):
        self.link = SklearnWrapperClassifier(links.Linear(10, 3))
        self.x = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)

    def check_invalid_argument(self):
        with self.assertRaises(AssertionError):
            x = chainer.Variable(self.link.xp.asarray(self.x))
            self.link(x)

    def test_invalid_argument_cpu(self):
        self.check_invalid_argument()

    @attr.gpu
    def test_invalid_argument_gpu(self):
        self.link.to_gpu()
        self.check_invalid_argument()

if __name__ == '__main__':
    unittest.main()
