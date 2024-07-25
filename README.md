# A Unified Differentiable Boolean Operator with Fuzzy Logic
<!-- <img src="./assets/teaser.jpg" width="100%"> -->

Public code release for "A Unified Differentiable Boolean Operator with Fuzzy Logic". For more details, please refer to:

**Surface Simplification using Intrinsic Error Metrics**<br>
[Hsueh-Ti Derek Liu](https://www.dgp.toronto.edu/~hsuehtil/), [Maneesh Agrawala](https://graphics.stanford.edu/~maneesh/), [Cem Yuksel](http://www.cemyuksel.com/), Tim Omernick, Vinith Misra, Stefano Corazza, [Morgan McGuire](https://casual-effects.com/), [Victor Zordan](https://people.computing.clemson.edu/~vbz/)<br>
SIGGRAPH 2024<br>
**[[Preprint](https://www.dgp.toronto.edu/~hsuehtil/pdf/fuzzyBoolean.pdf)]** **[[ArXiv](https://arxiv.org/abs/2407.10954)]**

## Installation
To get started, clone this repository 
``` bash
git clone https://github.com/HTDerekLiu/fuzzy-boolean.git
```
This code depends on [libigl](https://libigl.github.io/libigl-python-bindings/), [PyMCubes](https://github.com/pmneila/PyMCubes), [PyTorch](https://pytorch.org/) and some common python dependencies (e.g., NumPy). 

## Layout
Each folder contains stand-alone examples that demonstrate some core functionalities of our proposed technique. All of them have a similar directory and file layout:
+ [`00_product_fuzzy_boolean/`](00_product_fuzzy_boolean/): a minimal example to demonstrate how to perform boolean operations with the Product Fuzzy Logic. 
+ [`01_inverse_csg_2D/`](01_inverse_csg_2D/): performs inverse CSG optimization on a simple 2D example with an output CSG tree. 
+ [`02_inverse_csg_quadrics_3D/`](02_inverse_csg_quadrics_3D/): shows inverse CSG optimization on a single 3D shape.

And they share a common `utils` folder for basic functionalities.

## Execution
Running the code is very simple, one can simply do
``` bash
cd 00_product_fuzzy_boolean
python main.py
```

## Citation
If this code contributes to academic work, please cite as:
```bibtex
@inproceedings{Liu:2024:FuzzyBoolean,
    author = {Liu, Hsueh-Ti Derek and Agrawala, Maneesh and Yuksel, Cem and Omernick, Tim and Misra, Vinith and Corazza, Stefano and Mcguire, Morgan and Zordan, Victor},
    title = {A Unified Differentiable Boolean Operator with Fuzzy Logic},
    year = {2024},
    isbn = {9798400705250},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3641519.3657484},
    doi = {10.1145/3641519.3657484},
    booktitle = {ACM SIGGRAPH 2024 Conference Papers},
    articleno = {109},
    location = {Denver, CO, USA},
    series = {SIGGRAPH '24}
}
```