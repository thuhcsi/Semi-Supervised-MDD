#!/bin/bash

stage=0
stop_stage=2

PYTHON_ENVIRONMENT=semi
CONDA_ROOT=~/miniconda3

source ${CONDA_ROOT}/etc/profile.d/conda.sh

cwd=$(pwd)
FAIRSEQ=${cwd}/fairseq/fairseq
CODE=${cwd}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Install conda environment..."
    
    conda create --name ${PYTHON_ENVIRONMENT} python=3.8 -y
fi

conda activate ${PYTHON_ENVIRONMENT}

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Install fairseq and other dependencies..."
    cd ${cwd}/fairseq

    if [ $(pip freeze | grep fairseq | wc -l ) -gt 0 ]; then
        echo "Already installed fairseq. Skip..."
    else
        echo "Install fairseq..."
        python -m pip install --editable ./ -i https://pypi.tuna.tsinghua.edu.cn/simple
    fi
    # optionally do this if related error occurs
    python setup.py build_ext --inplace
    
    python -m pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
    python -m pip install soundfile -i https://pypi.tuna.tsinghua.edu.cn/simple
    python -m pip install praat-parselmouth -i https://pypi.tuna.tsinghua.edu.cn/simple
    python -m pip install tensorboardX -i https://pypi.tuna.tsinghua.edu.cn/simple
    python -m pip install datasets -i https://pypi.tuna.tsinghua.edu.cn/simple
    python -m pip install librosa -i https://pypi.tuna.tsinghua.edu.cn/simple
    python -m pip install editdistance -i https://pypi.tuna.tsinghua.edu.cn/simple
    python -m pip install seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install textgrid

    pip install git+https://github.com/sequitur-g2p/sequitur-g2p@master
    
    cd ${cwd}
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Copy model files to fairseq..."
    rsync -a Semi-Supervised-MDD/ fairseq/fairseq/
    
fi
