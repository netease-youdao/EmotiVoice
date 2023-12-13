import os

from setuptools import find_packages, setup

requirements={
    "infer": [
        "numpy>=1.24.3",
        "scipy>=1.10.1",
        "torch>=2.1",
        "torchaudio",
        "soundfile>=0.12.0",
        "librosa>=0.10.0",
        "scikit-learn",
        "numba==0.58.1",
        "inflect>=5.6.0",
        "tqdm>=4.64.1",
        "pyyaml>=6.0",
        "transformers==4.26.1",
        "yacs",
        "g2p_en",
        "jieba",
        "pypinyin",
        "streamlit",
        "pandas>=1.4,<2.0",
    ],
    "openai": [
        "fastapi",
        "python-multipart",
        "uvicorn[standard]",
        "pydub",
    ],
    "train": [
        "jsonlines",
        "praatio",
        "pyworld",
        "flake8",
        "flake8-bugbear",
        "flake8-comprehensions",
        "flake8-executable",
        "flake8-pyi",
        "mccabe",
        "pycodestyle",
        "pyflakes",
        "tensorboard",
        "einops",
        "matplotlib",
    ]
}

infer_requires = requirements["infer"]
openai_requires = requirements["infer"] + requirements["openai"]
train_requires = requirements["infer"] + requirements["train"]

VERSION = '0.2.0'

with open("README.md", "r", encoding="utf-8") as readme_file:
    README = readme_file.read()


setup(
    name="EmotiVoice",
    version=VERSION,
    url="https://github.com/netease-youdao/EmotiVoice",
    author="Huaxuan Wang",
    author_email="wanghx04@rd.netease.com",
    description="EmotiVoice 😊: a Multi-Voice and Prompt-Controlled TTS Engine",
    long_description=README,
    long_description_content_type="text/markdown",
    license="Apache Software License",
    # package
    packages=find_packages(),
    project_urls={
        "Documentation": "https://github.com/netease-youdao/EmotiVoice/wiki",
        "Tracker": "https://github.com/netease-youdao/EmotiVoice/issues",
        "Repository": "https://github.com/netease-youdao/EmotiVoice",
    },
    install_requires=infer_requires,
    extras_require={
        "train": train_requires,
        "openai": openai_requires,
    },
    python_requires=">=3.8.0",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)