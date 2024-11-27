ENVNAME="new_env"

conda create --name $ENVNAME && \
eval "$(conda shell.bash hook)" && \
conda activate $ENVNAME && \

# INSTALL CONDA PACKAGES
conda install -c conda-forge python -y && \

# INSTALL PIP PACKAGES
pip install -e .
