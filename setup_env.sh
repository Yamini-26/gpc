#!/bin/bash
pip install -e .
pip uninstall nvidia-cuda-nvcc-cu12 -y
rm -rf ~/miniconda3/envs/gpc/lib/python3.12/site-packages/nvidia/cuda_nvcc
echo "Environment ready!"
