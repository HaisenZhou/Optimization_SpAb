# Optimization_SpAb

## Overview
Optimization_SpAb is a computational framework for optimizing random heteropolypeptides as synthetic polyclonal antibodies. This project provides tools and methods for improving the binding affinity and selectivity of random heteropolypeptides through iterative optimization methods including Bayesian optimization and Genetic algorithm .

## Features
- Optimization of random heteropolypeptide compositions
- Synthetic polyclonal antibody design
- Optimization ensemble

## Requirements
- Python 3.7+
- Required packages:
  - NumPy
  - Pandas
  - Scikit-learn
  - BioPython
  - Pytorch

## Installation
```bash
git clone https://github.com/HaisenZhou/Optimization_SpAb.git
cd Optimization_SpAb
pip install -r requirements.txt
```

## Usage
1. **Data Preparation**
   - Prepare your HTS data files including All_data, Last_iteration, and Random_data.
   - Format input files with polymer compostion (n*8), Target protein ELISA results and Control protein ELISA results 

2. **Running Optimization**

Execute the optimization with:
```bash
Run run_optimization.py
```


## Contact
- **Author**: Haisen Zhou
- **Email**: [Zhouhaisen@pku.edu.cn]
- **GitHub**: [@HaisenZhou](https://github.com/HaisenZhou)

