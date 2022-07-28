from setuptools import setup, find_packages
setup(
    name='sprnn',
    version='1.0',
    packages=find_packages(['sprnn', 'sprnn.*'], exclude=['tests']),
    install_requires=[
        'natsort==7.1.1',
        'pandas==1.3.3',
        'pillow',
        'rsa==4.7.2',
        'scikit-learn==1.0.1',
        'scipy',
        'seaborn',
        'tensorboard==2.6.0',
        'tensorboard-data-server==0.6.1',
        'tensorboard-plugin-wit==1.8.0',
        'torch==1.8.1',
        'tqdm==4.61.2',
        'protobuf==3.19.4',
    ],
    license='MIT License',
    long_description=open('README.md').read(),
)