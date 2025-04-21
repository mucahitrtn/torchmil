from setuptools import setup, find_packages

setup(
    name="torchmil",  # Replace with your package name
    version="0.1.0",  # Your package version
    author="Francisco Miguel Castro-Macías",
    author_email="francastro8b@gmail.com",
    description="Multiple Instance Learning in PyTorch.",
    long_description=open("README.md").read(),  # Reads from a README file
    long_description_content_type="text/markdown",
    url="https://github.com/Franblueee/torchmil",  # Your repository URL
    packages=find_packages(),  # Automatically find and include packages
    install_requires=[
        "torch",
        "mkdocs",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Minimum Python version required
)