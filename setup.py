with open('README.rst', 'r') as f:
    readme = f.read()

with open('pyNNST.py', 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if line.startswith("__version__ ="):
            # Extract the version string between single quotes
            start = line.find("'")
            end = line.rfind("'")
            if start != -1 and end != -1 and end > start:
                __version__ =  line[start+1:end]
                break

#from distutils.core import setup, Extension
from setuptools import setup, Extension
# from pyNNST import __version__
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