from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="scgcl",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Single-cell Graph Contrastive Learning for cell type clustering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/scGCL",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "torch>=1.12.0",
        "torch-geometric>=2.1.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "umap-learn>=0.5.0",
        "tqdm>=4.62.0",
        "joblib>=1.1.0",
        "anndata>=0.8.0",
        "scanpy>=1.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "isort>=5.0.0",
        ],
    },
)
