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

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import argparse
import chainer
from chainer import serializers
from chainer_sklearn.links import SklearnWrapperClassifier

sys.path.append(os.pardir)
from mlp import MLP


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
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
    parser.add_argument('--example', '-ex', type=int, default=3,
                        help='Example mode')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    model = SklearnWrapperClassifier(MLP(args.unit, 10), device=args.gpu)

    if args.example == 1:
        print("Example 1. fit with x, y numpy array (same with sklearn's fit)")
        x, y = concat_examples(train)
        model.fit(x, y)
    elif args.example == 2:
        print("Example 2. Train with Chainer's dataset")
        # `train` is TupleDataset in this example
        # Even this one line work! (but no validation)
        model.fit(train)
    else:
        print("Example 3. Train with configuration")
        model.fit(
            train,
            test=test,
            batchsize=args.batchsize,
            #iterator_class=chainer.iterators.SerialIterator,
            optimizer=chainer.optimizers.Adam(),
            epoch=args.epoch,
            out=args.out,
            snapshot_frequency=1,
            #dump_graph=False
            #log_report=True,
            plot_report=False,
            #print_report=True,
            progress_report=False,
            resume=args.resume
        )

    # Save trained model
    serializers.save_npz('{}/mlp.model'.format(args.out), model)


if __name__ == '__main__':
    main()
