# Structure
# ########################
mkdir -p data/devmap
mkdir -p data/tc

# Pull dependencies
# ########################
pip install -r requirements.txt

BASE_DIR=$(pwd)

cd dependencies

# DeepTune
# ##########
git clone https://github.com/ChrisCummins/paper-end2end-dl.git
cd paper-end2end-dl
git checkout f9650454dfb33eea1ae8b7264a169dc02f01fb99
cd ..

# NCC
# ##########
git clone https://github.com/spcl/ncc.git
cd ncc
git checkout f9650454dfb33eea1ae8b7264a169dc02f01fb99

# Apply patch to e.g.
# - Fix requirements
# - Introduce grouped split
# - Write out raw data as result
git apply ../../patches/ncc/ncc.patch

# Add additional required data
cd published_results
wget https://polybox.ethz.ch/index.php/s/AWKd60qR63yViH8/download
unzip download
cd ..

cd ..

# Leave dependencies folder
cd ..


# Build native components
########################

# Clang extractor
# ##########
cd lib/c/clang_miner
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=$BASE_DIR/dependencies/llvm-project/build/lib/cmake/clang
make
cd ../../../..

# LLVM extractor
# ##########
cd lib/c/miner_llvm_pass
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=$BASE_DIR/dependencies/llvm-project/build/lib/cmake/llvm
make
cd ../../../..


# Environments
# ########################
mkdir -p envs

virtualenv envs/ncc --python=python3
envs/ncc/bin/pip install -r dependencies/ncc/requirements.txt
