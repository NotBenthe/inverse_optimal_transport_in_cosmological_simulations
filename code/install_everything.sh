#! /bin/bash

module purge
module load HDF5
module load Python/3.11.5-GCCcore-13.2.0

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install ipython

CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/easybuild/easybuild/el7/software/HDF5/1.10.5-gompi-2019b \
  pip install --no-binary=h5py h5py==3.11
 
pip install swiftsimio
pip install matplotlib healpy
pip install velociraptor

cd venv
git clone https://github.com/jchelly/VirgoDC.git
git clone https://github.com/jchelly/LightconeIO.git
pip install VirgoDC/python/
pip install LightconeIO/
cd ..
 
echo "To activate this evnironment run the following command(consider adding it to the end of your ~/.tcshrc file):"
# Note this assumes the user is using tcsh instead of bash
echo "$ module purge"
echo "$ module load HDF5"
echo "$ module load Python/3.11.5-GCCcore-13.2.0"
echo "$ source $(readlink -f venv/bin)/activate.csh"
