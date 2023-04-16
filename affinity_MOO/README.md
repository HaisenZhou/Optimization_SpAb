## Affnity_MOO
Install the miniconda:  
'''
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
bash Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
'''
Restart the terminal  
Install the conda environment:  
'''
conda env create -f env.yaml
'''

To install this module for development:  
'''
python setup.py develop
'''