from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="titanic-dataset-analysis",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Comprehensive machine learning analysis of the Titanic dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Titanic-Dataset",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "dash>=2.0.0",
        "dash-bootstrap-components>=1.0.0",
        "scipy>=1.7.0",
        "networkx>=2.6.0",
        "xgboost>=1.5.0",
        "lightgbm>=3.3.0",
        "pingouin>=0.5.0",
        "shap>=0.40.0",
        "jupyter>=1.0.0",
        "jupyterlab>=3.0.0",
        "joblib>=1.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.942",
        ],
    },
    entry_points={
        "console_scripts": [
            "titanic-predict=scripts.titanic_predictor:main",
            "titanic-dashboard=scripts.Titanic_Interactive_Dashboard:main",
        ],
    },
)