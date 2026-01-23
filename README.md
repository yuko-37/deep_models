conda env create -f environment.yml
conda activate deep
conda env update --name deep --file=./environment.yml --prune

# update conda itself
conda update -n base -c defaults conda

python -m pip install tensorflow
python -m pip install tensorflow-macos