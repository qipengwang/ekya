from setuptools import setup, find_packages

setup(
    name='ekya',
    version='0.0.1',
    url='https://github.com/romilbhardwaj/ekya.git',
    author='Romil Bhardwaj',
    author_email='romilb@eecs.berkeley.edu',
    description='Ekya - A system for online training',
    packages=find_packages(),
    install_requires=['ray', 'tensorflow', 'waymo-open-dataset',
                      'opencv-contrib-python', "tensorflow==2.2.0"],
)
