# coding: utf-8

import io
import os
import subprocess
from setuptools import find_packages, setup


# 包元信息
NAME = 'byte_mlperf'                                        # 实际包的名字
DESCRIPTION = 'Bytedance AI Accelerator Benchmark Tool'     # 项目描述
URL = 'https://github.com/bytedance/ByteMLPerf'             # 项目仓库 URL
EMAIL = 'jianzhe.xiao@bytedance.com'                        # 维护者邮箱地址
AUTHOR = '@data/aml/mlsys'                                  # 维护者姓名

# 项目运行需要的依赖
REQUIRES = [
    'matplotlib',
    'pandas>=1.3.5,<2.0',
    'virtualenv==16.7.9',
    'scikit-learn',
    'prompt_toolkit',
    'tqdm==4.62.3',
    'opencv-python',
    'transformers',
    'tokenization',
    'fpdf',
    'typing-extensions==3.7.4.3',
    'numpy==1.20.0',
]

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except IOError:
    long_description = DESCRIPTION


about = {}
with io.open(os.path.join(here, NAME, 'version.py')) as f:
    exec(f.read(), about)

setup(
    # add the 'byted' prefix for package name
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    # url=URL,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='boilerplate',
    packages=find_packages(),
    include_package_data = True,
    install_requires=REQUIRES,
    python_requires='>=3.6',
    dependency_links=[
        'https://pypi.byted.org/simple',
        'https://bytedpypi.byted.org/simple'
    ],
)
