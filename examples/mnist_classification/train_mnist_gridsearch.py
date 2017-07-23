#!/usr/bin/env python
"""
Originally from  
https://github.com/chainer/chainer/blob/master/examples/mnist/train_mnist.py

But SklearnWrapperClassifier fit method is used for training, 
instead of explicitly configure trainer.
"""
from __future__ import print_function
import os
import sys

from chainer.dataset import concat_examples
from sklearn import grid_search
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import argparse
import chainer
from chainer import serializers

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
from mlp import MLP
from SklearnWrapper import SklearnWrapperClassifier


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=5,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=50,
                        help='Number of units')
    parser.add_argument('--example', '-ex', type=int, default=1,
                        help='Example mode')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    if args.example == 1:
        print("Example 1. simple hyper parameter search")
        predictor = MLP(args.unit, 10)
        model = SklearnWrapperClassifier(predictor, device=args.gpu)
        optimizer1 = chainer.optimizers.SGD()
        optimizer2 = chainer.optimizers.Adam()
        gs = GridSearchCV(model,
                          {
                              # hyperparameter search for predictor
                              #'n_units': [10, 50],
                              # hyperparameter search for different optimizers
                              'optimizer': [optimizer1, optimizer2],
                              # 'batchsize', 'epoch' can be used as hyperparameter
                              'epoch': [args.epoch],
                              'batchsize': [100, 1000],
                          },
                          fit_params={
                              'progress_report': False,
                          }, verbose=2)
    elif args.example == 2:
        print("Example 2. search predictor's hyper parameter")
        predictor_constructor = MLP
        model = SklearnWrapperClassifier(predictor_constructor, device=args.gpu)
        optimizer = chainer.optimizers.Adam()
        gs = GridSearchCV(model,
                          {
                              # n_units is an argument for predictor_constructor
                              'n_units': [10, 50, 100],
                              # 'batchsize', 'epoch' can be used as hyperparameter
                              'batchsize': [100]
                          },
                          fit_params={
                              'epoch': args.epoch,
                              'optimizer': optimizer,
                              'progress_report': False,
                              #'test': test,
                              'out': args.out,
                              'snapshot_frequency': 1
                          }, verbose=2)
    elif args.example == 3:
        print("Example 3. search optimizer's hyper parameter")

        def build_mlp(n_units=100):
            return MLP(n_units, n_out=10)

        # You may also use function as predictor constructor
        predictor_constructor = build_mlp
        model = SklearnWrapperClassifier(predictor_constructor, device=args.gpu)
        optimizer_constructor = chainer.optimizers.Adam
        gs = GridSearchCV(model,
                          {
                              # n_units is an argument for predictor_constructor
                              'n_units': [10, 100],
                              # alpha is an argument for optimizer constructor
                              'alpha': [0.001, 0.0001],
                              'batchsize': [100],
                          },
                          fit_params={
                              'epoch': args.epoch,
                              'optimizer': optimizer_constructor,
                              'progress_report': False,
                              #'test': test,
                              'out': args.out,
                              'snapshot_frequency': 1
                          }, verbose=3)
    elif args.example == 4:
        print("Example 4. Randomized Search")

        model = SklearnWrapperClassifier(MLP, device=args.gpu)
        optimizer1 = chainer.optimizers.SGD()
        optimizer2 = chainer.optimizers.MomentumSGD()
        optimizer3 = chainer.optimizers.Adam()
        gs = RandomizedSearchCV(model,
                                {
                                    'n_units': [10, 50, 100, 500, 1000],
                                    # hyperparameter search for different optimizers
                                    'optimizer': [optimizer1, optimizer2, optimizer3],
                                    # 'batchsize', 'epoch' can be used as hyperparameter
                                    'batchsize': [10, 100, 500],
                                },
                                n_iter=5,
                                fit_params={
                                    'progress_report': False,
                                    'epoch': args.epoch
                                }, verbose=2)
    else:
        assert False, 'args.example took invalid value!'
    gs.fit(train)
    print('The parameters of the best model: {}'.format(gs.best_params_))

    best_model = gs.best_estimator_
    best_mlp = best_model.predictor

    # Save trained model
    serializers.save_npz('{}/best_mlp.model'.format(args.out), best_mlp)

if __name__ == '__main__':
    main()
