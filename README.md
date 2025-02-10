# Optimization_SpAb

## Overview
**Optimization_SpAb** is a computational framework designed to optimize random heteropolypeptides (RHPs) as synthetic polyclonal antibodies (SpAbs). This project provides tools and methods for enhancing the binding affinity and selectivity of RHPs through iterative optimization techniques, including **Bayesian optimization** and **genetic algorithms**. The approach enables rapid identification of SpAbs for specific protein targets, with potential applications in diagnostics and therapeutics.

## Features
- **Data-driven optimization** of random heteropolypeptide compositions
- **Synthetic polyclonal antibody (SpAb) design** for selective antigen recognition
- **Integration of Bayesian Optimization (BO) and Genetic Algorithm (GA)** for enhanced discovery
- **High-throughput data processing and screening**
- **Flexible scoring system** for affinity and selectivity tuning

## Requirements
- **Python** 3.7+
- Required dependencies:
  - NumPy
  - Pandas
  - Scikit-learn
  - BioPython
  - PyTorch

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/HaisenZhou/Optimization_SpAb.git
cd Optimization_SpAb
pip install -r requirements.txt
```

## Usage
### 1. **Data Preparation**
Ensure your dataset contains the following files:
- **All_data.csv**: The complete dataset including all experimental results
- **Last_iteration.csv**: Data from the previous optimization cycle
- **Random_data.csv**: All random sampling data for optimization

The input data should be formatted as:
- Polymer composition (n × 8 matrix, where n is the number of samples)
- Target protein ELISA results
- Control protein ELISA results

### 2. **Running Optimization**
Run the optimization script:
```bash
python run_optimization.py
```
This script will execute Bayesian Optimization and Genetic Algorithm iterations to refine SpAb compositions.

### 3. **Output & Analysis**
The optimization results will be saved in the output directory

## Contact
- **Author**: Haisen Zhou
- **Email**: [Zhouhaisen@pku.edu.cn]
- **GitHub**: [@HaisenZhou](https://github.com/HaisenZhou)


For more details, please refer to our research article: *Data-driven Design of Random Heteropolypeptides as Synthetic Polyclonal Antibodies*.
