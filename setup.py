from setuptools import setup, find_packages

setup(
    name="condensed-matter-ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "pytest>=6.2.0",
        "jupyter>=1.0.0",
        "aifeynman>=2.0.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for analyzing condensed matter physics using AIFeynman",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/condensed-matter-ai",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 