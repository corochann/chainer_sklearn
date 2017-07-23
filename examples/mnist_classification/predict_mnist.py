"""Inference/predict code for MNIST

model must be trained before inference, train_mnist_4_trainer.py must be executed beforehand.
"""
from __future__ import print_function
import os
import sys

import argparse
import numpy as np

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
from SklearnWrapper import SklearnWrapperClassifier


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--modelpath', '-m', default='result/mlp.model',
                        help='Model path to be loaded')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=50,
                        help='Number of units')
    args = parser.parse_args()

    batchsize = 128

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    # Load trained model
    model = SklearnWrapperClassifier(MLP(args.unit, 10), device=args.gpu)
    serializers.load_npz(args.modelpath, model)

    # --- Example 1. Predict all test data ---
    outputs = model.predict(test,
                            batchsize=batchsize,
                            retain_inputs=True,)

    x, t = model.inputs
    y = outputs
    #y = outputs[0]

    #print(type(y), len(y))
    #print(y.shape)
    #print(len(model.inputs))
    #print(x.shape, t.shape)

    # --- check all the results ---
    wrong_count = np.sum(y != t)
    print('wrong inference {}/{}'.format(wrong_count, len(test)))

    # --- Example 2. Predict partial test data ---
    outputs = model.predict_proba(test[:20])
    x, t = model.inputs
    #y, = outputs
    y = outputs

    # --- Plot result ---
    """Original code referenced from https://github.com/hido/chainer-handson"""
    ROW = 4
    COLUMN = 5
    # show graphical results of first 20 data to understand what's going on in inference stage
    plt.figure(figsize=(15, 10))
    for i in range(ROW * COLUMN):
        np.set_printoptions(precision=2, suppress=True)
        print('{}-th image: answer = {}, predict = {}'.format(i, t[i], F.softmax(y[i:i+1]).data))
        example = (x[i] * 255).astype(np.int32).reshape(28, 28)
        plt.subplot(ROW, COLUMN, i+1)
        plt.imshow(example, cmap='gray')
        plt.title("No.{0} / Answer:{1}, Predict:{2}".format(i, t[i], np.argmax(y[i], axis=0)))
        plt.axis("off")
    plt.tight_layout()
    plt.savefig('predict.png')


if __name__ == '__main__':
    main()
