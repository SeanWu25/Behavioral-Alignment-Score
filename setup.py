from setuptools import setup, find_packages

setup(
    name="bas_eval",  
    version="0.1.0",
    packages=find_packages(), 
    install_requires=[
        "numpy",
        "pandas",
    ],
    author="Sean Wu",
    description="A decision-theoretic framework for measuring LLM Alignment and Uncertainty",
)