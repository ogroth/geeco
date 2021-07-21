#!/bin/bash

# prerequisistes
# (mini)conda: https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# MuJoCo 2.0 installed under ${HOME}/.mujoco/mujoco200

# paths
export GEECO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# create conda environment
CONDA_ENV_NAME="geeco"
conda env create -f ${GEECO_ROOT}/environment.yml
conda activate ${CONDA_ENV_NAME}

# register environment variables
mkdir -p ${CONDA_PREFIX}/etc/conda/activate.d
mkdir -p ${CONDA_PREFIX}/etc/conda/deactivate.d
echo "
#!/bin/sh
export GEECO_ROOT=${GEECO_ROOT}
export PYTHONPATH=${GEECO_ROOT}/src
" > ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh
echo "
#!/bin/sh
unset GEECO_ROOT
unset PYTHONPATH
" > ${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh