from setuptools import setup, find_packages

setup(
    name="phase3",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "xgboost",
        "lightgbm",
        "scikit-learn",
        "pandas",
    ],
)
