from setuptools import setup, find_packages

setup(
    name="sevennet-inference",
    version="0.1.0",
    description="SevenNet inference package for MOF property predictions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="SevenNet Inference Contributors",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9,<3.12",
    install_requires=[
        "sevenn>=0.9.0",
        "ase>=3.22.0",
        "numpy>=1.21.0,<2.0.0",
        "scipy>=1.7.0",
        "torch>=2.0.0",
        "phonopy>=2.20.0",
        "pymatgen>=2023.0.0",
        "h5py",
        "matplotlib>=3.5.0",
        "pandas",
        "tqdm",
        "prettytable",
        "pyyaml",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "sevennet-infer=sevennet_inference.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
