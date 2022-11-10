#conda create -n gdn_old python=3.7.0
#conda activate gdn_old

pip install --find-links https://download.pytorch.org/whl/torch_stable.html torch==1.5.0+cpu

pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.5.0%2Bcpu.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.5.0%2Bcpu.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.5.0%2Bcpu.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.5.0%2Bcpu.html
pip install torch-geometric==1.5.0

## TO RUN
## python main.py -dataset msl -device cpu
## python main.py -dataset simriver -device cpu