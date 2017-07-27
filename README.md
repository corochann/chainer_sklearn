# Chainer-sklearn-wrapper
Experiment implementation to support sklearn like interface in Chainer


## Supported interface

 - `fit` function to train the model, it can be used by following 2 ways:
   - Train with conventional sklearn way, `model.fit(train_x, train_y)`.
   - Train with `train` dataset of Chainer dataset class  by `model.fit(train)`.
     
     So that we can also train the model with own dataset (`DatasetMixin` class etc).

 - `predict` method to predict the classify result for classifier, 
 predict the actual value for regressor.

   It can be used like `answer[i:j] = model.predict(X[i:j])`, 
   without considering about batchsize for loop.

 - `predict_proba` method to predict the probability of each category (only for classifier). 

 - `score` method to calculate how much is the model prediction score.

### `fit` function for training

See `examples/train_mnist_fit.py` and try

`python train_mnist_fit.py --ex example_id -g gpu_id`


You can write training code as follows,
```angular2html
    train, test = chainer.datasets.get_mnist()

    model = SklearnWrapperClassifier(MLP(args.unit, 10))    
    model.fit(
        train,
        test=test,
        batchsize=args.batchsize,
        iterator_class=chainer.iterators.SerialIterator,
        optimizer=chainer.optimizers.Adam(),
        device=args.gpu,
        epoch=args.epoch,
        out=args.out,
        snapshot_frequency=1,
        dump_graph=False
        log_report=True,
        plot_report=True,
        print_report=True,
        progress_report=True,
        resume=args.resume
    )
```

instead of conventional way (configuring `trainer` explicitly),
```angular2html
    model = L.Classifier(MLP(args.unit, 10))
    ...
    
    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    trainer.extend(extensions.dump_graph('main/loss'))

    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    trainer.extend(extensions.LogReport())

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

```


### `GridSearchCV`, `RandomizedSearchCV` support

You can execute hyper parameter search using sklearn's `GridSearchCV` or 
`RandomizedSearchCV` class.

See `examples/mnist_classification/train_mnist_gridsearch.py` for the example.

Try

`python train_mnist_gridsearch.py --ex example_id -g gpu_id`
