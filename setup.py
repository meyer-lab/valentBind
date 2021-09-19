from setuptools import setup, find_packages

setup(name='valentBind',
      version='0.1',
      description='The multivalent binding model implemented in Python.',
      url='https://github.com/meyer-lab/valentBind',
      license='MIT',
      packages=find_packages(exclude=['doc']),
      install_requires=['numpy', 'scipy'])