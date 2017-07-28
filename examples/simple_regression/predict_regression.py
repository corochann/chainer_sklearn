"""Inference/predict code for MNIST

model must be trained before inference, train_mnist_4_trainer.py must be executed beforehand.
"""
from __future__ import print_function
import os
import sys

import argparse
import numpy as np

from train_regression_fit import load_data

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F
from chainer import serializers

from mlp import MLP
sys.path.append(os.pardir)
from SklearnWrapper import SklearnWrapperClassifier, SklearnWrapperRegressor


def main():
    parser = argparse.ArgumentParser(description='Regression predict')
    parser.add_argument('--modelpath', '-m', default='result/mlp.model',
                        help='Model path to be loaded')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=50,
                        help='Number of units')
    args = parser.parse_args()

    batchsize = 128

    # Load dataset
    data, target = load_data()
    X = data.reshape((-1, 1)).astype(np.float32)
    y = target.reshape((-1, 1)).astype(np.float32)

    # Load trained model
    model = SklearnWrapperRegressor(MLP(args.unit, 1), device=args.gpu)
    serializers.load_npz(args.modelpath, model)

    # --- Example 1. Predict all test data ---
    outputs = model.predict(X,
                            batchsize=batchsize,
                            retain_inputs=False,)

    # --- Plot result ---
    plt.figure()
    plt.scatter(X, y, label='actual')
    plt.plot(X, outputs, label='predict', color='red')
    plt.legend()
    plt.show()
    plt.savefig('predict.png')


if __name__ == '__main__':
    main()
