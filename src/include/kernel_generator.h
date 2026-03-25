#pragma once
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include "config.h"

/* Since this is for the Gemma model, the kernels to be made are specific to the Gemma model. This requires a Linear layer (basically matrix multiplication), a SwiGLU (Swish + GLU)
 an embedding layer, a RoPE layer, an attention block (along with MQA optims) and an RMS norm layer. A lot more info is present in the implementation file. */


class KernelGenerator {
public:
    KernelGenerator(llvm::LLVMContext& Ctx, llvm::Module* M, const GemmaConfig& Config);

    llvm::Function* emitRMSNorm();
    llvm::Function* emitLinear(const std::string& name);
    llvm::Function* emitSwiGLU(int size, const std::string& name);
    llvm::Function* emitRoPE(int head_dim, const std::string& name);
    llvm::Function* emitSoftmax(const std::string& name);
    llvm::Function* emitKVCacheUpdate(int kv_dim, const std::string& name);
    llvm::Function* emitAttentionScore(int head_dim, const std::string& name);
    llvm::Function* emitAttentionValueSum(int head_dim, const std::string& name);
    llvm::Function* emitEmbeddingLookup(int hidden_size, const std::string& name);

private:
    llvm::LLVMContext&  Ctx;
    llvm::Module*       M;
    const GemmaConfig&  Config;
    llvm::IRBuilder<>   Builder;
};