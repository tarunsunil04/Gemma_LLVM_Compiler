#include <llvm/Support/TargetSelect.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/Transforms/Utils/Mem2Reg.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>

#include <llvm/Transforms/Utils/PromoteMemToReg.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>

#include "include/jit.h"


/* This portion is responsible for initializing the JIT backend that will do a lot of the opitm and heavy lifting for us. The LLVM API allows us to do things that are pretty 
bare metal. */

using namespace llvm;
using namespace llvm::orc;

GemmaJIT::GemmaJIT() {
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();

    auto JTMB = JITTargetMachineBuilder::detectHost();

    auto JITExpect = LLJITBuilder()
        .setJITTargetMachineBuilder(std::move(*JTMB))
        .setNumCompileThreads(4)
        .create();

    if (!JITExpect) {
        throw std::runtime_error("Failed to create LLJIT instance");
    }
    JIT = std::move(*JITExpect);

    GlobalContext = std::make_unique<LLVMContext>();
}

//this function is reposnsible for the speed up of the input kernel. This enables a buch of optimization options that drastically speed up everything in the kernel.  
void GemmaJIT::optimizeModule(llvm::Module& M) {
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;

    PassBuilder PB;
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    FunctionPassManager FPM = PB.buildFunctionSimplificationPipeline(
        OptimizationLevel::O2, ThinOrFullLTOPhase::None);

    //this is the magic pass that turns alloca into registers/PHIs which are PARAMOUNT for speedy ahh math
    FPM.addPass(PromotePass());

    for (auto& F : M) {
        if (!F.isDeclaration()) { //Only optimize functions with declrs
            FPM.run(F, FAM);
        }
    }
}

void GemmaJIT::addModule(std::unique_ptr<Module> M, std::unique_ptr<LLVMContext> C) {
    ThreadSafeModule TSM(std::move(M), std::move(C));
    auto Err = JIT->addIRModule(std::move(TSM));
    if (Err) {
        throw std::runtime_error("Failed to add IR Module to JIT");
    }
}

void* GemmaJIT::getFunctionPtr(const std::string& Name) {
    auto Sym = JIT->lookup(Name);
    if (!Sym) {
        throw std::runtime_error("Could not find function: " + Name);
    }
    return reinterpret_cast<void*>(static_cast<uint64_t>(Sym->getValue()));
}

GemmaJIT::~GemmaJIT() {
    printf("Cleanliness is next to godliness");
}