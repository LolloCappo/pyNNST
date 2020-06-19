with open('README.rst', 'r') as f:
    readme = f.read()
    
def parse_requirements(filename):
    ''' Load requirements from a pip requirements file '''
    with open(filename, 'r') as fd:
        lines = []
        for line in fd:
            line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines

requirements = parse_requirements('requirements.txt')

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