cd dependencies

# LLVM
# ##########
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout release/7.x

# Build
mkdir build
cd build
cmake ../llvm -DLLVM_ENABLE_PROJECTS=clang
make -j$(nproc) libclang libllvm
make -j$(nproc) opt
make -j$(nproc) clang
make -j$(nproc) clang-format
cd ../..
