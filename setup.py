from distutils.core import setup

from setuptools import find_packages

setup_requires = ['numpy']
install_requires = [
    'scikit-learn',
    'chainer>=2.0',
]

setup(name='chainer_sklearn',
      version='0.0.1',
      description='Chainer scikit-learn wrapper.',
      packages=find_packages(),
      author='corochann',
      author_email='corochannz@gmail.com',
      url='https://github.com/corochann/chainer-sklearn-wrapper',
      license='MIT',
      setup_requires=setup_requires,
      install_requires=install_requires
)
