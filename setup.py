from setuptools import setup, find_packages

setup(
    name="morsegraph",
    version="0.1.0",
    author="Bernardo Rivas",
    author_email="bernardo.dopradorivas@utoledo.edu",
    description="A lightweight library for Morse graph analysis of dynamical systems.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "networkx",
        "matplotlib",
    ],
    extras_require={
        'ml': ['torch']
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
