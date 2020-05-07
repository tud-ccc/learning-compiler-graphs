# list source files here
set(miner_src miner.cc funcinfo.cc)

add_library(miner_pass MODULE
    ${miner_src}
)

# Use C++11 to compile your pass (i.e., supply -std=c++11).
target_compile_features(miner_pass PRIVATE
    cxx_range_for cxx_auto_type
)

# LLVM is (typically) built with no C++ RTTI. We need to match that;
# otherwise, we'll get linker errors about missing RTTI data.
set_target_properties(miner_pass PROPERTIES
    COMPILE_FLAGS "-fno-rtti -Wall -Wextra"
)

# Get proper shared-library behavior (where symbols are not necessarily
# resolved when the shared library is linked) on OS X.
if(APPLE)
    set_target_properties(miner_pass PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup"
    )
endif(APPLE)


add_executable(miner
    main.cc ${miner_src}
)

llvm_map_components_to_libnames(REQ_LLVM_LIBRARIES ${LLVM_TARGETS_TO_BUILD}
    asmparser core linker bitreader irreader ipo scalaropts
    analysis support
)

target_link_libraries(miner ${REQ_LLVM_LIBRARIES})

set_target_properties(miner PROPERTIES
    LINKER_LANGUAGE CXX
    PREFIX ""
    COMPILE_FLAGS "-fno-rtti -Wall -Wextra"
)

target_compile_features(miner PRIVATE
    cxx_range_for cxx_auto_type
)
