cmake_minimum_required(VERSION 3.20)
project(GemmaLLVMCompiler LANGUAGES CXX)

include(FetchContent)

set(LLVM_DIR "C:/Program Files (x86)/LLVM/lib/cmake/llvm")
find_package(LLVM REQUIRED CONFIG HINTS ${LLVM_DIR})
find_package(OpenMP REQUIRED)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake from: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(third_party/sentencepiece)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_MakeAvailable(json)

add_executable(GemmaCompiler 
    src/main.cpp 
    src/jit_engine.cpp
    src/model_loader.cpp
    "src/include/model_loader.h" 
    "src/include/config.h" 
    "src/include/jit.h" 
    "src/include/kernel_generator.h" 
    "src/kernel_generator.cpp" 
    "src/include/kv_cache.h" 
    "src/include/gemma_tokenizer.h" 
    "src/gemma_engine.cpp" 
    "src/include/gemma_engine.h" 
    "src/gemma_tokenizer.cpp" 
)

target_compile_features(GemmaCompiler PRIVATE cxx_std_20)

if(MSVC)
    set_target_properties(GemmaCompiler PROPERTIES LINK_FLAGS "/STACK:10000000") #Just in case
    target_compile_options(GemmaCompiler PRIVATE "/GS-")
    target_compile_options(GemmaCompiler PRIVATE "/bigobj")
    
    set_target_properties(GemmaCompiler PROPERTIES LINK_FLAGS "/NODEFAULTLIB:zlib.lib")
    set(MSVC_SYS_LIBS shell32 advapi32 psapi user32 version shlwapi)
endif()

llvm_map_components_to_libnames(llvm_libs 
    core 
    orcjit 
    native 
    support 
    target
)

target_link_libraries(GemmaCompiler 
    PRIVATE 
    ${llvm_libs} 
    ${MSVC_SYS_LIBS} 
    nlohmann_json::nlohmann_json 
    sentencepiece-static
    OpenMP::OpenMP_CXX
)

target_include_directories(GemmaCompiler PRIVATE third_party/sentencepiece/src)