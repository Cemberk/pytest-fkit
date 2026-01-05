"""Setup for pytest-fkit."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pytest-fkit",
    version="0.1.0",
    author="Cemberk",
    description="A pytest plugin that prevents crashes from killing your test suite, with execution tracing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Framework :: Pytest",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pytest>=6.0.0",
    ],
    extras_require={
        "tracer": ["psutil>=5.0.0"],
        "pysr": ["pysr>=0.16.0", "numpy>=1.20.0", "pandas>=1.3.0"],
        "all": ["psutil>=5.0.0", "pysr>=0.16.0", "numpy>=1.20.0", "pandas>=1.3.0"],
    },
    entry_points={
        "pytest11": [
            "fkit = pytest_fkit.plugin",
            "fkit_tracer = pytest_fkit.tracer.pytest_plugin",
        ],
        "console_scripts": [
            "fkit-tracer = pytest_fkit.tracer.cli:main",
        ],
    },
)
