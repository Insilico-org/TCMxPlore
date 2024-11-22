# TCMxPlore

TCMxPlore is a Python library for exploring and analyzing Traditional Chinese Medicine (TCM) databases, enabling connections between modern pharmacology and traditional herbal medicine. It provides tools for querying, analyzing, and cross-referencing TCM herbs, formulas, and their molecular targets.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- Access and analyze data from BATMAN-TCM 2.0 and DragonTCM databases
- Cross-reference herbs and formulas between databases
- Search for compounds and herbs based on protein targets
- Design custom formulas using compound-target associations
- Analyze herb-formula-condition relationships
- Generate formula recommendations based on molecular signatures

## Installation

1. Clone the repository:
```bash
git clone https://github.com/f-galkin/TCMxPlore.git
cd TCMxPlore
```

2. Create and activate conda environment:
```bash
conda env create -f environment.yml
conda activate tcm_env
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

```python
from tcmxplore.connector import BatmanDragonConnector

# Load the pre-trained connector with BATMAN-TCM and DragonTCM data
connector = BatmanDragonConnector.from_huggingface()

# Access individual databases
batman_db = connector.dbs['BATMAN']
dragon_db = connector.dbs['American Dragon']

# Find herbs targeting specific compounds
compound_ids = [445154, 5280343]  # resveratrol and quercetin
herbs = batman_db.select_N_herbs_by_cids(
    cids=compound_ids,
    min_cids=1,
    N_top=5
)

# Look up formula annotations in DragonTCM
formulas = dragon_db.select_N_formulas_by_herbs(
    herbs=["REN SHEN", "HUANG QI"],
    min_herbs=2,
    N_top=3
)
```

## Core Modules

### batman_db

The `batman_db` module interfaces with BATMAN-TCM 2.0, providing access to:

- Over 17,000 known compound-target interactions 
- ~2.3 million predicted compound-target interactions
- Natural compounds and their PubChem IDs
- Herb chemical compositions
- Traditional formulas 

```python
from tcmxplore.batman_db import TCMDB

# Load BATMAN-TCM database
db = TCMDB.from_huggingface()

# Find compounds targeting specific genes
compounds = db.find_enriched_cpds(
    gene_list=["TNF", "IL6", "NFKB1"],
    tg_type='both',
    thr=0.01
)

# Create custom formula
formula = db.get_greedy_formula(compound_ids, max_herb=4)
```

### dragon_db

The `dragon_db` module interfaces with DragonTCM database containing:

- 1,119 TCM conditions mapped to SNOMED
- 2,580 traditional formulas with compositions
- 1,044 herbs with detailed annotations
- 28,735 entity relationships

```python
from tcmxplore.dragon_db import TCMAnnotationDB

# Load DragonTCM database
db = TCMAnnotationDB.from_huggingface()

# Get herb details
herb = db.herbs["REN SHEN"]
print(herb.properties)
print(herb.indications)

# Find formulas for specific conditions
formulas = db.select_N_formulas_by_herbs(
    herbs=["REN SHEN", "HUANG QI"],
    min_herbs=2
)
```

### connector

The `connector` module enables cross-referencing between BATMAN-TCM and DragonTCM databases:

```python
from tcmxplore.connector import BatmanDragonConnector

# Initialize connector
connector = BatmanDragonConnector.from_huggingface()

# Search across databases
results = connector.match_total(
    query="ginseng",
    N=5,
    thr=75
)

# Find equivalent herbs across databases
connector.get_equivalent_herbs('BATMAN', 'American Dragon')
```

## Use Cases

### Drug Discovery
Using BATMAN-TCM's extensive compound-target interaction data (17,000+ known and 2.3M predicted interactions), you can:

1. Screen TCM herbs for compounds targeting specific proteins:
   - Query compounds by protein targets using Fisher's exact test
   - Filter compounds based on significance thresholds
   - Map compounds to their source herbs

2. Design novel formulas based on molecular mechanisms:
   - Use greedy algorithms to optimize herb combinations
   - Ensure coverage of desired protein targets
   - Balance traditional composition principles with molecular evidence

3. Identify therapeutic compounds:
   - Search by protein targets relevant to specific conditions
   - Cross-reference with DragonTCM's clinical annotations
   - Validate against traditional usage patterns

### TCM Research
- Analyze traditional formula compositions
- Study herb-formula-condition relationships
- Cross-reference between different TCM sources

### Clinical Applications
- Search for formulas treating specific conditions
- Understand herb properties and interactions
- Access detailed annotations and usage guidelines

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Data Sources

This library uses data from:

1. BATMAN-TCM 2.0 ([Kong et al., 2024](https://doi.org/10.1093/nar/gkad926))
2. DragonTCM dataset based on American Dragon (www.americandragon.com) and works of Dr. Joel Penner

## Citation

If you use TCMxPlore in your research, please cite both the library and the underlying databases:

```bibtex
@software{tcmxplore2024,
  title = {TCMxPlore: A Python Library for Traditional Chinese Medicine Analysis},
  author = {Galkin, Fedor},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/f-galkin/TCMxPlore}
}

@article{10.1093/nar/gkad926,
    author = {Kong, Xiangren and Liu, Chao and Zhang, Zuzhen and others},
    title = "{BATMAN-TCM 2.0: an enhanced integrative database for known and predicted interactions between traditional Chinese medicine ingredients and target proteins}",
    journal = {Nucleic Acids Research},
    volume = {52},
    number = {D1},
    pages = {D1110-D1120},
    year = {2023},
    doi = {10.1093/nar/gkad926}
}

@misc{galkin_dragontcm_2024,
    author = {{Fedor Galkin}},
    title = {DragonTCM (Revision 5b535a4)},
    year = 2024,
    url = {https://huggingface.co/datasets/f-galkin/DragonTCM},
    doi = {10.57967/hf/3557},
    publisher = {Hugging Face}
}

@misc {fedor_galkin_2024,
	author       = { {Fedor Galkin} },
	title        = { batman2 (Revision 88429db) },
	year         = 2024,
	url          = { https://huggingface.co/datasets/f-galkin/batman2 },
	doi          = { 10.57967/hf/3314 },
	publisher    = { Hugging Face }
}

@book{mcdonald_zang_1994,
    title = {Zang Fu Syndromes: Differential Diagnosis and Treatment},
    isbn = {978-0-9650529-0-0},
    publisher = {Lone Wolf Press},
    author = {McDonald, John and Penner, Joel},
    year = {1994}
}

```

