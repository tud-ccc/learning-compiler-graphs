cmake_minimum_required(VERSION 3.0)
project(clang_miner)

set(CMAKE_CXX_STANDARD 14)
add_definitions("-fno-exceptions -fno-rtti")

find_package(Clang REQUIRED)
find_package(LLVM REQUIRED)

add_executable(clang_miner clang_miner.cpp)
target_link_libraries(clang_miner
        clang
        clangAST
        clangTooling
        clangFrontendTool)
target_include_directories(clang_miner
        PRIVATE ${CLANG_INCLUDE_DIRS}
                ${LLVM_INCLUDE_DIRS})
