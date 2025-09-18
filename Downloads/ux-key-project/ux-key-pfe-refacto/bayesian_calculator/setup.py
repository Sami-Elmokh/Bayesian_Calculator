from setuptools import setup, find_packages

setup(
    name='bayesian_calculator',
    version='1.0',
    packages=find_packages(),
    description=' Estimate the ergonomy of a website ',
    long_description=open('README.md').read(),
    author='EL Mejdani && Sakli',
    author_email='mohamed.elmejdani@student-cs.fr',
    license='MIT',
    install_requires=[
        # list your package dependencies
    ],
    url='https://github.com/yourusername/mylibrary',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
