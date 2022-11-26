import io
from setuptools import setup, find_packages

with io.open('./README.md', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='gense',
    packages=['gense'],
    version='0.1',
    license='MIT',
    description='A sentence embedding tool based on GenSE',
    author='Yiming Chen, Yan Zhang, Bin Wang, Zuozhu Liu, Haizhou Li',
    author_email='yiming.chen@u.nus.edu',
    url='https://github.com/MatthewCYM/GenSE',
    keywords=['sentence', 'embedding', 'gense', 'nlp'],
    install_requires=[
        "tqdm",
        "scikit-learn",
        "scipy>=1.5.4,<1.6",
        "transformers==4.15.0",
        "torch==1.10.1",
        "numpy>=1.19.5,<1.20",
        "setuptools",
        "datasets",
        "pandas",
        "prettytable",
    ]
)