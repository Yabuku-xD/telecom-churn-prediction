"""
Setup script for Telecom Customer Churn Prediction System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="telecom-churn-prediction",
    version="1.0.0",
    author="Data Science Team",
    author_email="datascience@company.com",
    description="End-to-end customer churn prediction system for telecommunications companies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/company/telecom-churn-prediction",
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "churn-predict=src.predictor:main",
            "churn-train=src.model_trainer:main",
            "churn-dashboard=app.streamlit_dashboard:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.csv"],
    },
    project_urls={
        "Bug Reports": "https://github.com/company/telecom-churn-prediction/issues",
        "Source": "https://github.com/company/telecom-churn-prediction",
        "Documentation": "https://telecom-churn-prediction.readthedocs.io/",
    },
)