import os
import sys
from pathlib import Path
from setuptools import setup, find_packages, find_namespace_packages


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


common_setup_kwargs = {
    "name": "vllm_mixin",
    "version": "0.0.2",
    "author": "Qianle Wang",
    "author_email": "wql20000111@stu.sufe.edu.cn",
    "description": "An easy mixin with just a single line of code to seamlessly integrate powerful functionalities into VLLM.",
    "long_description": readme(),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/wangitu/vllm-mixin",
    "keywords": ["vllm", "LLM-serving", "large-language-models", "mixin"],
    "platforms": ["windows", "linux"],
    "classifiers": [
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
}


requirements = [
    "vllm"
]

extras_require = {
    "auto_gptq": ["AutoGPTQ", "unpadded-AutoGPTQ"]
}

setup(
    packages=find_namespace_packages(exclude=["assets"]),
    package_data={"": ["*.yaml", "*.txt"]},
    install_requires=requirements,
    extras_require=extras_require,
    python_requires=">=3.8",
    **common_setup_kwargs
)
