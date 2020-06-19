with open('README.rst', 'r') as f:
    readme = f.read()
    

#from distutils.core import setup, Extension
from setuptools import setup, Extension
from pyNNST import __version__
setup(name='pyNNST',
      version=__version__,
      author='Lorenzo Capponi',
      author_email='lorenzocapponi@outlook.it',
      description='Definition of non-stationary index for time-series',
      url='https://github.com/LolloCappo/pyNNST',
      py_modules=['pyNNST'],
      long_description=readme,
      install_requires='numpy'
      )