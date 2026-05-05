# Welcome to CSSlib 👋
![License](https://img.shields.io/github/license/AIRI-Institute/CSSlib?style=flat&logo=opensourceinitiative&logoColor=white&color=blue) ![Version](https://img.shields.io/badge/version-1.2-orange.svg?cacheSeconds=2592000)

<p align="center">
  <img src="https://raw.githubusercontent.com/AIRI-Institute/CSSlib/refs/heads/main/logo.jpg" width="30%" title="CSSlib" alt="CSSlib"/>
</p>

CSSlib is an open-source code for building configuration search space (CSS) of disordered crystals, loading of the CSS dataset obtained, local/remote MPI or SLURM calculations and data visualization.

## Table of content
- [Installation](#installation)
- [Contributors](#contributors)
- [Tutorial](#tutorial)
- [References & Citing](#references--citing)

## Installation
**CSSlib** can be installed through 
1) the **pip** package manager (in the virtual environment):
```sh
pip install csslib
```
2) the **git clone** command:
```sh
git clone https://github.com/AIRI-Institute/CSSlib.git CSSlib
cd CSSlib
pip install .
```
3) the **uv** package manager:
```sh
uv venv .venv
source .venv/bin/activate
uv pip install csslib
```

**CSSlib** by default requires **Supercell** program. Details on **Supercell** installation can be found at the corresponding [website](https://orex.github.io/supercell/download/).

As a calculator for quantum mechanical calculations, **CSSlib** assumes the use of the [VASP](https://www.vasp.at/) (Vienna Ab initio Simulation Package) software package. Also there is an optional dependency for the [QuantumEspresso](http://www.quantum-espresso.org) simulation package which can be installed after the normal installation as:
```sh
pip install --group espresso .
```
or 
```sh
uv sync --group espresso
```

## Contributors
- Aleksey Krautsou
- Aleksandr Solovykh

## Tutorial
The best way to learn how to use **CSSlib** is through the tutorial notebook located at the [tests directory](tests/csslib_example.ipynb) or at the [google collab](https://colab.research.google.com/drive/1OzYe7QXuL-BYXROnEh7hvuIW08xGlZO2?usp=sharing).

## References & Citing
If you use this code, please consider citing works that actively used the CSS approach, which resulted in the creation of this library:

1. A.V. Krautsou, I.S. Humonen, V.D. Lazarev, R.A. Eremin, S.A. Budennyy<br/>
   "Impact of crystal structure symmetry in training datasets on GNN-based energy assessments for chemically disordered CsPbI<sub>3</sub>"<br/>
   https://doi.org/10.1038/s41598-025-92669-3
2. N.A. Matsokin, R.A. Eremin, A.A. Kuznetsova, I.S. Humonen, A.V. Krautsou, V.D. Lazarev, Y.Z. Vassilyeva, A.Y. Pak, S.A. Budennyy, A.G. Kvashnin, A.A. Osiptsov<br/>
   "Discovery of chemically modified higher tungsten boride by means of hybrid GNN/DFT approach"<br/>
   https://doi.org/10.1038/s41524-025-01628-z
3. R.A. Zaripov, R.A. Eremin, I.S. Humonen, A.V. Krautsou, V.V. Kuznetsov, K.E. GermanS, S.A. Budennyy, S.V. Levchenko</br>
   "First-principles data-driven approach for assessment of stability of Tc-C systems"</br>
   https://doi.org/10.1016/j.actamat.2025.121704
