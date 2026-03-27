# Gemma LLVM Compiler 

Exactly what it sounds like: An LLVM-based compiler for the Gemma 2B Model. The best way to describe this project would be to compare it to `gemma.cpp`. This is basically `gemma.cpp`, but written using the LLVM API, as opposed to 
Google's Highway SIMD API.

Note that for now, this works with Gemma-2B-it with BF16 precision only. Support for quantized models with other file formats (as of now, only SAFETENSORS are supported) are also yet to come.

With BF16 precision, this compiler achieves a TFTT of ~4 seconds (for simple prompts, larger and specific prompts may go upwards of 50 seconds. After all this is a 2B parameter model!) and a TPS (Tokens Per Second) of ~6. (Tested locally on a laptop with 32GB RAM and 8GB VRAM. 🥀)

Hopefully the TPS and TFTT improves with quantized models!

## Setup

### Disclaimer

  * Make sure you have LLVM. [Click here if you don't have it.](https://llvm.org/docs/GettingStarted.html)
  * If you don't have it in your PATH, you're gonna have to change the `set(LLVM_DIR "C:/Program Files (x86)/LLVM/lib/cmake/llvm")` line in the `CMakeLists.txt` file to the directory of choice.
  * I made this project using VS 2022, but that being said, this is fashioned as a CMake project. Goes without saying that you need CMake for this. [Click here if you don't have it.](https://cmake.org/download/)
  * OpenMP is also used for parallelizing the whole thing. Most compilers have it by default, so I wouldn't worry about this that much, but if you're feeling adventurous, you can possibly replace this with CUDA kernels or something fancier.
  * The `sentencepiece` library (for the tokenizer logic) is also used here. So is the `JSON` library. ([This one to be exact.](https://github.com/nlohmann/json))  

### Setup 

  * Clone this repo.

    ```(bash)
      git clone https://github.com/tarunsunil04/Gemma_LLVM_Compiler.git
    ```
  * Generate the build files associated with the project.

    ```(bash)
      mkdir build && cd build && cmake ..
    ```
  * Build the project. (Remove the `--config` argument if you want no particular config mode. But this will almost always guarantee slower performance.)

    ```(bash)
      cmake --build build --config Release
    ```

  * Run the executable marked as `GemmaCompiler.exe`. (Will be in `../out/build/x64-Release/`, or similar.)

    
    
