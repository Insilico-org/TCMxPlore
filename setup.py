from setuptools import setup, find_packages

# Read version from package __init__.py
with open("tcmxplore/__init__.py", "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

setup(
    name="tcmxplore",
    version=version,
    description="A framework for discovering anti-aging TCM formulas with bioinformatics and generative AI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Fedor Galkin",
    author_email="f.galkin@insilico.com",
    url="https://github.com/insilicomedicine/TCMxPlore",
    packages=find_packages(include=["tcmxplore", "tcmxplore.*"]),
    install_requires=[
        "pandas",
        "pyarrow",
        "scipy",
        "thefuzz",
        "numpy>=1.24",
        "tqdm",
        "typing-extensions>=4.0",
        "pyyaml>=5.1",
        "requests>=2.19",
        "fsspec>=2023.1",
        "aiohttp",
        "packaging>=20.0",
        "filelock",
        "suffix-tree",
        "just-agents-core==0.4.3",
        "chembl_webresource_client>=0.10",
        "datasets>=2.14",
        "huggingface-hub>=0.17",
        "json-repair>=0.5.0"
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'ipykernel>=6.0.0'
        ],
        'notebooks': [
            'jupyter>=1.0.0',
            'matplotlib>=3.7.0',
            'seaborn>=0.12.0'
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License", 
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires=">=3.11",
    include_package_data=True,
    package_data={
        'tcmxplore': ['data/*.json', 'data/*.gz'],
    }
)
