"""

Some of the implementations referenced from keras scikit wrapper implementation.
https://github.com/fchollet/keras/blob/7f58b6fbe702c1936e88a878002ee6e9c469bc77/keras/wrappers/scikit_learn.py
"""
import inspect
import types
import copy

import numpy
from chainer.datasets import DictDataset, ImageDataset, LabeledImageDataset, TupleDataset
from chainer.functions import mean_squared_error

from sklearn.base import BaseEstimator, ClassifierMixin

import chainer
from chainer.dataset import concat_examples, DatasetMixin
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link, Optimizer, cuda
from chainer import training
from chainer.training import extensions
from chainer import reporter


import chainer.datasets


def is_function(obj):
    """Check if obj is function(lambda function or user defined method) or not"""
    return isinstance(obj, (types.FunctionType, types.MethodType,
                            types.LambdaType))


def is_dataset(obj):
    """Check if obj is Chainer dataset instance or not"""
    return isinstance(obj, (DictDataset, ImageDataset, LabeledImageDataset,
                            TupleDataset, DatasetMixin))


class SklearnBaseWrapper(link.Chain):
    """A simple classifier model.

    This is an example of chain that wraps another chain. It computes the
    loss and accuracy based on a given input/label pair.

    Args:
        predictor (~chainer.Link): Predictor network.
        lossfun (function): Loss function.
        accfun (function): Function that computes accuracy.

    Attributes:
        predictor (~chainer.Link): Predictor network or build predictor function.
        lossfun (function): Loss function.
        accfun (function): Function that computes accuracy.
        y (~chainer.Variable): Prediction for the last minibatch.
        loss (~chainer.Variable): Loss value for the last minibatch.
        accuracy (~chainer.Variable): Accuracy for the last minibatch.
        compute_accuracy (bool): If ``True``, compute accuracy on the forward
            computation. The default value is ``True``.

    """
    # [Note] Setting _estimator_type to classifier changes the behavior of
    # `check_cv` for cross validation, to force set the label `y` when creating
    # cross validation dataset.
    # Which is problematic when we want to use Chainer dataset instead of (X, y) notation

    # _estimator_type = "classifier"  # For sklearn compatibility
    # __name__ = 'tmp'

    def __init__(self,
                 predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy,
                 device=-1,
                 **sk_params
                 ):
        super(SklearnBaseWrapper, self).__init__()
        if isinstance(predictor, chainer.Link):
            # print('[DEBUG] predictor instance')
            with self.init_scope():
                self.predictor = predictor
            self.predictor_constructor = predictor.__class__
        elif is_function(predictor) or issubclass(predictor, chainer.Link):
            # print('[DEBUG] predictor is constructor')
            self.predictor_constructor = predictor
        else:
            print("[ERROR] predictor should be either Chain class instance or"
                  "function which returns Chain class instance")
            assert False

        self.lossfun = lossfun
        self.accfun = accfun
        self.compute_accuracy = accfun is not None
        self.y = None
        self.loss = None
        self.accuracy = None
        self.inputs = None

        # Ensure initialization, necessary for GridSearch
        self.predictor.to_cpu()
        self.device = -1
        self.update_device(device)

        self.sk_params = sk_params

    def update_device(self, device=None):
        if device >= 0:
            chainer.cuda.get_device(
                device).use()  # Make a specified GPU current
            self.to_gpu(device)  # Copy the model to the GPU
        else:
            self.to_cpu()  # Copy the model to the CPU
        self.device = device

    def build(self):
        """build predictor

        This function is used when `predictor` is not set at `__init__`.
        """
        if self.predictor_constructor is None:
            print('[ERROR] build_predictor_fn not set, skip.')
        else:
            if hasattr(self, 'predictor'):
                print(
                    "[WARNING] predictor is already set, predictor is overridden")
                del self.predictor
            with self.init_scope():
                self.predictor = self.predictor_constructor(
                    **self.filter_sk_params(self.predictor_constructor)
                )
        self.update_device(self.device)

    def __call__(self, *args):
        """Computes the loss value for an input and label pair.

        It also computes accuracy and stores it to the attribute.

        Args:
            args (list of ~chainer.Variable): Input minibatch.

        The all elements of ``args`` but last one are features and
        the last element corresponds to ground truth labels.
        It feeds features to the predictor and compare the result
        with ground truth labels.

        Returns:
            ~chainer.Variable: Loss value.

        """
        assert len(args) >= 2
        x = args[:-1]
        t = args[-1]
        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(*x)
        self.loss = self.lossfun(self.y, t)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss

    def _forward(self, *args):
        """Predicts by the model's output by returning `predictor`'s output

        """
        if self.predictor is None:
            print("[ERROR] predictor is not set or not build yet.")
            return
        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                return self.predictor(*args)

    def forward_batch(self, *args, batchsize=16, retain_inputs=True,
                      calc_score=False):
        """
        Accuracy is used for score when self.accuracy is True,
        otherwise, `loss` is used for score calculation.
        
        :param args: 
        :param batchsize: 
        :param retain_inputs: 
        :param calc_score: 
        :return: 
        """
        data = args[0]

        input_list = None
        output_list = None
        total_score = 0
        for i in range(0, len(data), batchsize):
            inputs = concat_examples(data[i:i + batchsize], device=self.device)
            if not isinstance(inputs, tuple):
                inputs = (inputs,)
            # print('inputs', inputs, len(inputs))
            outputs = self._forward(*inputs)
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            # Init
            if retain_inputs:
                if input_list is None:
                    input_list = [[] for _ in range(len(inputs))]
                for j, input in enumerate(inputs):
                    input_list[j].append(cuda.to_cpu(input))  # TODO: review
            if output_list is None:
                output_list = [[] for _ in range(len(outputs))]
            for j, output in enumerate(outputs):
                # print(j, 'output', type(output), output.shape)
                output_list[j].append(cuda.to_cpu(output.data))  # TODO: review
            if calc_score:  # TODO: switch accuracy or loss depends on situation.
                if self.compute_accuracy:
                    total_score += self.accuracy * outputs[0].shape[0]
                else:
                    total_score += self.loss * outputs[0].shape[0]

        if retain_inputs:
            self.inputs = [numpy.concatenate(input) for input in input_list]
        if calc_score:
            self.total_score = cuda.to_cpu(total_score.data) / len(data)

        result = [numpy.concatenate(output) for output in output_list]
        if len(result) == 1:
            return result[0]
        else:
            return result

    def predict_log_proba(self, X):
        pass

    def decision_function(self, X):
        pass

    def transform(self, X):
        pass

    def inverse_transform(self, X):
        pass

    def fit(self, *args, **kwargs):
        """If hyper parameters are set to None, then instance's variable is used,
        this functionality is used Grid search with `set_params` method.
        Also if instance's variable is not set, _default_hyperparam is used. 

        Usage: model.fit(train_dataset) or model.fit(X, y)

        Args:
            train: training dataset, assumes chainer's dataset class 
            test: test dataset for evaluation, assumes chainer's dataset class
            batchsize: batchsize for both training and evaluation
            iterator_class: iterator class used for this training, 
                            currently assumes SerialIterator or MultiProcessIterator
            optimizer: optimizer instance to update parameter
            epoch: training epoch
            out: directory path to save the result
            snapshot_frequency (int): snapshot frequency in epoch. 
                                Negative value indicates not to take snapshot.
            dump_graph: Save computational graph info or not, default is False.
            log_report: Enable LogReport or not
            plot_report: Enable PlotReport or not
            print_report: Enable PrintReport or not
            progress_report: Enable ProgressReport or not
            resume: specify trainer saved path to resume training.

        """
        kwargs = self.filter_sk_params(self.fit_core, kwargs)
        return self.fit_core(*args, **kwargs)

    def fit_core(self, *args,
                 test=None,
                 batchsize=16,
                 epoch=10,
                 optimizer=None,
                 iterator_class=chainer.iterators.SerialIterator,
                 out='result',
                 snapshot_frequency=-1,
                 dump_graph=False,
                 log_report=True,
                 plot_report=True,
                 print_report=True,
                 entries=None,
                 progress_report=True,
                 resume=None,
                 extensions_list=None,
                 **kargs
                 ):
        if len(args) == 1:
            train = args[0]
        elif len(args) >= 2:
            if len(args) >= 3:
                print('[WARNING]: assuming args as (x, Y)')
            train = chainer.datasets.TupleDataset(*args)
        else:
            print('[ERROR]: train data not specified')
            assert False

        # Construct predictor if necessary
        predictor_kwargs = self.filter_sk_params(self.predictor_constructor)
        if len(predictor_kwargs) > 0:
            self.build()

        if not hasattr(self, 'predictor'):
            assert False, 'predictor is not build yet'

        # Construct optimizer if necessary
        if optimizer is None:
            _optimizer = chainer.optimizers.SGD()
        elif isinstance(optimizer, Optimizer):
            optimizer_constructor = optimizer.__class__
            optimizer_kwargs = self.filter_sk_params(optimizer_constructor)
            if len(optimizer_kwargs) == 0:
                _optimizer = optimizer
            else:
                _optimizer = optimizer_constructor(**optimizer_kwargs)
        elif is_function(optimizer) or issubclass(optimizer, Optimizer):
            # `optimizer` is constructor of optimizer
            _optimizer = optimizer(**self.filter_sk_params(optimizer))
        else:
            print('[ERROR] invalid optimizer passed')
            assert False

        # --- fit main code---
        # TODO: currently iterator_class assumes SerialIterator or MultiProcessIterator.
        train_iter = iterator_class(train, batchsize)
        test_iter = None
        if test is not None:
            test_iter = iterator_class(test, batchsize, repeat=False,
                                       shuffle=False)
        # Optimizer
        _optimizer.setup(self)

        # Set up a trainer
        updater = training.StandardUpdater(train_iter, _optimizer,
                                           device=self.device)
        trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

        if test_iter is not None:
            # Evaluate the model with the test dataset for each epoch
            trainer.extend(
                extensions.Evaluator(test_iter, self, device=self.device))

        if dump_graph:
            trainer.extend(extensions.dump_graph('main/loss'))

        if snapshot_frequency > 0:
            trainer.extend(extensions.snapshot(),
                           trigger=(snapshot_frequency, 'epoch'))

        if log_report:
            trainer.extend(extensions.LogReport())

        # Save two plot images to the result dir
        if plot_report and extensions.PlotReport.available():
            trainer.extend(
                extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                      'epoch', file_name='loss.png'))
            if self.compute_accuracy:
                trainer.extend(
                    extensions.PlotReport(
                        ['main/accuracy', 'validation/main/accuracy'],
                        'epoch', file_name='accuracy.png'))

        if print_report:
            if not entries:
                val = (test_iter is not None)
                entries = self._infer_entries(self.compute_accuracy, val)
            trainer.extend(extensions.PrintReport(entries))

        if progress_report:
            trainer.extend(extensions.ProgressBar())

        if extensions_list:
            for ext in extensions_list:
                trainer.extend(ext)

        if resume:
            # Resume from a snapshot
            chainer.serializers.load_npz(resume, trainer)
        # Run the training
        trainer.run()
        return self

    def _infer_entries(self, compute_accuracy, compute_validation):
        if compute_accuracy and compute_validation:
            entries = ['epoch', 'main/loss', 'validation/main/loss',
                       'main/accuracy', 'validation/main/accuracy',
                       'elapsed_time']
        elif compute_accuracy and not compute_validation:
            entries = ['epoch', 'main/loss', 'main/accuracy', 'elapsed_time']
        elif not compute_accuracy and compute_validation:
            entries = ['epoch', 'main/loss', 'validation/main/loss',
                       'elapsed_time']
        else:
            entries = ['epoch', 'main/loss', 'elapsed_time']
        return entries

    def score(self, *args, **kwargs):
        kwargs = self.filter_sk_params(self.score_core, kwargs)
        return self.score_core(*args, **kwargs)

    def score_core(self, *args, sample_weight=None, batchsize=16):
        # during GridSearch, which only assumes score(X, y) interface.
        if len(args) == 1:
            test = args[0]
            if isinstance(test, (numpy.ndarray)):  # TODO: reivew
                test = chainer.datasets.TupleDataset(test)
        elif len(args) >= 2:
            if len(args) >= 3:
                print('WARNING: assuming args as (x, Y)')
            test = chainer.datasets.TupleDataset(*args)
        else:
            assert False, 'ERROR: train data not specified'

        # For Classifier
        # `accuracy` is calculated as score, using `forward_batch`
        # For regressor
        # `loss` is calculated as score, using `forward_batch`
        self.forward_batch(test, batchsize=batchsize, retain_inputs=False, calc_score=True)
        return self.total_score

    def get_params(self, deep=True):
        """get_params is used to clone this estimator"""
        res = copy.deepcopy(self.sk_params)
        res.update({
            'lossfun': self.lossfun,
            'accfun': self.accfun,
            'device': self.device,
        })
        if hasattr(self, 'predictor'):
            res.update({'predictor': self.predictor})
        else:
            res.update({'predictor': self.predictor_constructor})
        return res

    def set_params(self, **parameters):
        """set_params is used to set Grid parameters"""
        for parameter, value in parameters.items():
            if parameter == 'predictor':
                if isinstance(value, chainer.Link):
                    del self.predictor
                    with self.init_scope():
                        self.predictor = value
                else:
                    assert False, 'predictor is not Chain instance'
            elif parameter in ['lossfun', 'accfun', 'device']:
                setattr(self, parameter, value)
            else:
                self.sk_params.update({parameter: value})
        return self

    def filter_sk_params(self, fn, override=None):
        override = override or {}
        res = {}
        sig = inspect.signature(fn)
        for name, value in self.sk_params.items():
            if name in sig.parameters:
                res.update({name: value})
        res.update(override)
        return res


class SklearnWrapperClassifier(SklearnBaseWrapper):

    """A simple classifier model.

    This is an example of chain that wraps another chain. It computes the
    loss and accuracy based on a given input/label pair.

    Args:
        predictor (~chainer.Link): Predictor network.
        lossfun (function): Loss function.
        accfun (function): Function that computes accuracy.

    Attributes:
        predictor (~chainer.Link): Predictor network or build predictor function.
        lossfun (function): Loss function.
        accfun (function): Function that computes accuracy.
        y (~chainer.Variable): Prediction for the last minibatch.
        loss (~chainer.Variable): Loss value for the last minibatch.
        accuracy (~chainer.Variable): Accuracy for the last minibatch.
        compute_accuracy (bool): If ``True``, compute accuracy on the forward
            computation. The default value is ``True``.

    """
    # [Note] Setting _estimator_type to classifier changes the behavior of
    # `check_cv` for cross validation, to force set the label `y` when creating
    # cross validation dataset.
    # Which is problematic when we want to use Chainer dataset instead of (X, y) notation

    #_estimator_type = "classifier"  # For sklearn compatibility
    #__name__ = 'tmp'

    def __init__(self,
                 predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy,
                 device=-1,
                 **sk_params
                 ):
        super(SklearnWrapperClassifier, self).__init__(
            predictor,
            lossfun=lossfun,
            accfun=accfun,
            device=device,
            **sk_params
        )

    def predict(self, *args, batchsize=16, retain_inputs=True):
        """predict the output

        Args:
            *args: input
            batchsize: batchsize to execute predict 
            retain_inputs: if True, inputs is saved to self.inputs 

        Returns: outputs of the model prediction (calculated by `predictor`)

        """
        proba = self.predict_proba(*args, batchsize=batchsize, retain_inputs=retain_inputs)
        if isinstance(proba, tuple):
            # TODO: review. If output of `predict_proba` is multiple,
            # use first output as probability array
            proba = proba[0]
        return numpy.argmax(proba, axis=1)

    def predict_proba(self, *args, batchsize=16, retain_inputs=True):
        """predict the output

        Args:
            *args: input
            batchsize: batchsize to execute predict 
            retain_inputs: if True, inputs is saved to self.inputs 
        Returns: outputs of the model prediction (calculated by `predictor`)

        """
        return self.forward_batch(*args, batchsize=batchsize, retain_inputs=retain_inputs)

    def predict_log_proba(self, X):
        pass

    def decision_function(self, X):
        pass

    def transform(self, X):
        pass

    def inverse_transform(self, X):
        pass


class SklearnWrapperRegressor(SklearnBaseWrapper):
    def __init__(self,
                 predictor,
                 lossfun=mean_squared_error,
                 accfun=None,
                 device=-1,
                 **sk_params
                 ):
        super(SklearnWrapperRegressor, self).__init__(
            predictor,
            lossfun=lossfun,
            accfun=accfun,
            device=device,
            **sk_params
        )

    def predict(self, *args, batchsize=16, retain_inputs=True):
        """predict the output

        Args:
            *args: input
            batchsize: batchsize to execute predict 
            retain_inputs: if True, inputs is saved to self.inputs 

        Returns: outputs of the model prediction (calculated by `predictor`)

        """
        return self.forward_batch(*args, batchsize=batchsize,
                                  retain_inputs=retain_inputs)

    def transform(self, X):
        pass

    def inverse_transform(self, X):
        pass

