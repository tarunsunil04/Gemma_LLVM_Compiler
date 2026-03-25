#pragma once

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include <memory>
#include <string>

/* This class is responsible for the intialization of the LLVM context, and passing in the kernels that we made, so that LLVM can do its thing, and make it fast as hell. */

class GemmaJIT {
public:
    GemmaJIT();
    ~GemmaJIT();

    void addModule(std::unique_ptr<llvm::Module> M, std::unique_ptr<llvm::LLVMContext> C);
    void* getFunctionPtr(const std::string& Name);
    llvm::LLVMContext& getGlobalContext() { return *GlobalContext; }
    void optimizeModule(llvm::Module& M);

private:
    std::unique_ptr<llvm::orc::LLJIT> JIT;
    std::unique_ptr<llvm::LLVMContext> GlobalContext;
};