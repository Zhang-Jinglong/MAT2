# MAT<sup>2</sup>

**M**anifold **A**lignment of Single-Cell **T**ranscriptomes with Cell **T**riplets

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)

## Overview

MAT<sup>2</sup> is designed to align multiple single-cell transcriptome datasets. The operation steps include:
1. **Manifold alignment based on contrastive learning**: For a cell of interest *C*, MAT<sup>2</sup> will select a cell *C<sub>p</sub>* from the same cell type but a different dataset and a cell *C<sub>n</sub>* from a different cell type to form a cell triplet (*C*, *C<sub>p</sub>*, *C<sub>n</sub>*). With contrastive learning, the distance between *C* and *C<sub>p</sub>* in the latent manifold space will be much smaller than that between *C* and *C<sub>n</sub>*, so as to achieve the alignment of single-cell transcriptome.
2. **Reconstruction of gene expression profile**: With neural network decoders, consensus gene expression and batch-specific deviation will be reconstructed. Among them, consensus gene expression can be used for downstream analysis such as differential expression analysis and lineage tracing.

## Installation

Firstly, please use git to clone the MAT<sup>2</sup> repository.

```
git clone https://github.com/Zhang-Jinglong/MAT2.git
cd MAT2/
```

The Python packages that MAT<sup>2</sup> depends on can be installed through conda. Run **setup.py** on the command line to install MAT<sup>2</sup>.
```
conda install --file requirements.txt --yes
python setup.py install
```

## Usage

There is an example jupyter notebook [`demo/test.ipynb`](demo/test.ipynb) in the source code of MAT<sup>2</sup>, which demonstrates the method of aligning single-cell transcriptome datasets using MAT<sup>2</sup>.

The following is a brief description of the usage of MAT<sup>2</sup> in Python:

### Loading datasets

The test data can be found in the [`demo/`](demo/) folder in the MAT<sup>2</sup> repository.

```Python
import pandas as pd
from MAT2 import *

# MAT2 receives pandas DataFrame as input data.
# Multiple batches of data are concated into a matrix of size gene_num * cell_num.
data = pd.read_csv('data.csv', header=0, index_col=0)

# The row name of metadata should correspond to the cell name in data.
# Metadata must contain the 'batch' column, and must also contain the 'type' column when supervised.
metadata = pd.read_csv('metadata.csv', header=0, index_col=0)

# Anchor needs to be loaded only in unsupervised situations.
# Each record contains two cell numbers (cell in [0,cell_num-1]) and a score (score in [0.0,1.0]).
anchor = pd.read_csv('anchor.csv', header=0, index_col=0)
```

### Building model & training

When providing cell type annotations for model building:

```Python
model = BuildMAT2(
    data=data,
    metadata=metadata,
    num_workers=2,
    use_gpu=True,
    mode='supervised',
    dropout_rate=0.3)
model.train(epochs=30)
```

When there is no cell type annotation but anchor is provided:

```Python
model = BuildMAT2(
    data=data,
    metadata=metadata,
    anchor=anchor,
    num_workers=2,
    use_gpu=True,
    mode='manual')
model.train(epochs=30)
```

When providing part of cell type annotations, run in semi-supervised mode:

```Python
model = BuildMAT2(
    data=data,
    metadata=metadata,
    anchor=anchor,
    num_workers=2,
    use_gpu=True,
    mode='semi-supervised')
model.train(epochs=30)
```

### Testing

```Python
# test_data = data
# Calculate the reconstructed consensus gene expression.
rec = model.evaluate(test_data)
# Your own downstream analysis.
```

## Citation

 @article{
    title={MAT2: manifold alignment of single-cell transcriptomes with cell triplets}, 
    volume={37}, 
    ISSN={1367-4803}, 
    DOI={10.1093/bioinformatics/btab250}, 
    number={19}, 
    journal={Bioinformatics}, 
    author={Zhang, Jinglong and Zhang, Xu and Wang, Ying and Zeng, Feng and Zhao, Xing-Ming}, 
    year={2021}, 
    pages={3263–3269} 
 }
