![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![pytest](https://github.com/lucaslehnert/rlutils/workflows/pytest/badge.svg)

# Reward-Predictive State Representations Generalize Across Task

This repository contains the implementations to reproduce the simulation results of [Lehnert et al. 2020][paper].
Please view the individual jupyter notebook files for instructions on how to reproduce all results.

## Obtaining Datasets and Reproduction Using Docker



## Installation and Running Code

To run the jupyter notebooks and reproduce the simulations, first clone the repository and install the required 
dependencies:

```
git clone https://github.com/lucaslehnert/rewardpredictive.git
cd rewardpredictive
pip install -r requirements.txt
```

The jupyter notebooks can be viewed by running 

```
jupyter notebook
```

from the project's root directory (the directory this README file is stored in).
All simulations can be reproduced by running the main script from the root directory:

```
python -m main -e [ExperimentName]
```

[paper]: https://www.biorxiv.org/content/10.1101/653493v2
