#include "include/kernel_generator.h"
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Verifier.h>


/* Now we're gettinng into the meat and potatoes of this thing. This is where the optimized mathematical operations (that are useful for an LLM anyway) are defined. 
Ultimately by using the LLVM API, a lot of the optimization comes easily (or more accurately, automatically) and is quite beneficial when it comes to running this on a
device that is supported by LLVM's backend. See here for the list of supported architectures: https://llvm.org/docs/CompilerWriterInfo.html#hardware */

/* As seen before, passing values by pointer is kind of the norm here. This has to do with the fact that a lot of these 'kernels' operate on raw memory buffers. So 
creating copies are out of question. */

/* Moreover, if you plan to write any kernels yourself, this resource helped me out A LOT: https://llvm.org/docs/WritingAnLLVMBackend.html */

/* With that boring preamble, let's jump neck-first into the volcano!*/

using namespace llvm;

//Using initializer lits are apparently faster.

KernelGenerator::KernelGenerator(LLVMContext& C, Module* Mod, const GemmaConfig& Conf)
    : Ctx(C), M(Mod), Config(Conf), Builder(C) {}


//This Function here is for the Gemma-style RMS norm. for other functions too, you can refer to the same link. (see here : https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/layernorm/) 

/* So this means it has */

Function* KernelGenerator::emitRMSNorm() 
{
    Type* floatTy   = Builder.getFloatTy();
    auto* vecTy     = FixedVectorType::get(floatTy, 8);

    FunctionType* FT    = FunctionType::get(Builder.getVoidTy(), { Builder.getPtrTy(), Builder.getPtrTy(), Builder.getPtrTy() }, false);
    Function* F         = Function::Create(FT, Function::ExternalLinkage, "gemma_rmsnorm", M);

    //The blocks involved
    BasicBlock* Entry       = BasicBlock::Create(Ctx, "entry", F);
    BasicBlock* SumLoop     = BasicBlock::Create(Ctx, "sum_loop", F);
    BasicBlock* ExitSum     = BasicBlock::Create(Ctx, "exit_sum", F);
    BasicBlock* ScaleLoop   = BasicBlock::Create(Ctx, "scale_loop", F);
    BasicBlock* Done        = BasicBlock::Create(Ctx, "done", F);

    Builder.setFastMathFlags(FastMathFlags::getFast());

    Builder.SetInsertPoint(Entry);
    Value* Input        = F->getArg(0);
    Value* Weight       = F->getArg(1);
    Value* Output       = F->getArg(2);

    /* Sum of Squares ova here */
    Value* ZeroVec  = ConstantAggregateZero::get(vecTy);
    Builder.CreateBr(SumLoop);
    Builder.SetInsertPoint(SumLoop);

    PHINode* IV         = Builder.CreatePHI(Builder.getInt32Ty(), 2, "i");
    IV->addIncoming(Builder.getInt32(0), Entry);

    PHINode* Accum      = Builder.CreatePHI(vecTy, 2, "accum");
    Accum->addIncoming(ZeroVec, Entry);

    Value* InPtr        = Builder.CreateGEP(floatTy, Input, IV);
    Value* InVec        = Builder.CreateLoad(vecTy, InPtr);

    Value* Sq           = Builder.CreateFMul(InVec, InVec);
    Value* NextAccum    = Builder.CreateFAdd(Accum, Sq);

    Value* NextIV       = Builder.CreateAdd(IV, Builder.getInt32(8));
    IV->addIncoming(NextIV, SumLoop);
    Accum->addIncoming(NextAccum, SumLoop);

    Value* Cond         = Builder.CreateICmpULT(NextIV, Builder.getInt32(Config.hidden_size));
    Builder.CreateCondBr(Cond, SumLoop, ExitSum);

    /* Scalar Math(i.e Squishing the vector, finding the square root, and then its reciprocal, followed by splatting(or broadcasting) into vectors) */
    Builder.SetInsertPoint(ExitSum);
    //Horizontal reduce: sum all lanes of the vector
    Value* FinalSum = Builder.CreateFAddReduce(ConstantFP::get(floatTy, 0.0), NextAccum);

    Value* Mean = Builder.CreateFDiv(FinalSum, ConstantFP::get(floatTy, (float)Config.hidden_size)); //<--- This can be done in a better way. Remeber, Division = "Not good".
    Value* MeanEps = Builder.CreateFAdd(Mean, ConstantFP::get(floatTy, Config.rms_norm_eps));

    Function* SqrtFn = Intrinsic::getOrInsertDeclaration(M, Intrinsic::sqrt, { floatTy });
    Value* Root = Builder.CreateCall(SqrtFn, { MeanEps });
    Value* InvRoot = Builder.CreateFDiv(ConstantFP::get(floatTy, 1.0f), Root);  //<--- This can be done in a better way. Remeber, Division = "Not good".

    //Splat InvRoot scalar into a vector for the scaling loop
    Value* InvRootVec = Builder.CreateVectorSplat(8, InvRoot);

    Builder.CreateBr(ScaleLoop);

    /* Scaling Loop, actual normalization takes place over here. */
    Builder.SetInsertPoint(ScaleLoop);
    PHINode* IV2 = Builder.CreatePHI(Builder.getInt32Ty(), 2, "j");
    IV2->addIncoming(Builder.getInt32(0), ExitSum);

    //Load Input (FP32)
    Value* S_InPtr = Builder.CreateGEP(floatTy, Input, IV2);
    Value* S_InVec = Builder.CreateAlignedLoad(vecTy, S_InPtr, MaybeAlign(32)); //Input is aligned h_state

    //Load Weight (We'll convert them into FP32 for efficent vector ops)
    Type* i16Ty = Builder.getInt16Ty();
    Value* S_WPtr = Builder.CreateGEP(i16Ty, Weight, IV2); //Scale by 2 bytes
    Value* WVecI16 = Builder.CreateLoad(FixedVectorType::get(i16Ty, 8), S_WPtr); //Unaligned load from mapped file

    //BF16 to FP32 Bit-shift (This kind of defeats the purpose of using BF16 precision in the first place, as we use FP32 registers for faster math, but hey, you've saved around 5 GB in memory!)
    Value* WVecI32 = Builder.CreateZExt(WVecI16, FixedVectorType::get(Builder.getInt32Ty(), 8));
    Value* WVecShifted = Builder.CreateShl(WVecI32, ConstantVector::getSplat(ElementCount::getFixed(8), Builder.getInt32(16)));
    Value* S_WVec = Builder.CreateBitCast(WVecShifted, vecTy);

    //out = (in * invRoot) * (1 + weight)
    Value* Normed = Builder.CreateFMul(S_InVec, InvRootVec);
    Value* OneVec = ConstantVector::getSplat(ElementCount::getFixed(8), ConstantFP::get(floatTy, 1.0f));
    Value* WPlusOne = Builder.CreateFAdd(S_WVec, OneVec);
    Value* Result = Builder.CreateFMul(Normed, WPlusOne);

    //Store the result
    Value* OutPtr = Builder.CreateGEP(floatTy, Output, IV2);
    Builder.CreateAlignedStore(Result, OutPtr, MaybeAlign(32));

    Value* NextIV2 = Builder.CreateAdd(IV2, Builder.getInt32(8));
    IV2->addIncoming(NextIV2, ScaleLoop);

    Value* Cond2 = Builder.CreateICmpULT(NextIV2, Builder.getInt32(Config.hidden_size));
    Builder.CreateCondBr(Cond2, ScaleLoop, Done);

    Builder.SetInsertPoint(Done);
    Builder.CreateRetVoid();

    if (verifyFunction(*F, &errs())) 
    {
        F->print(errs());
        throw std::runtime_error("Function verification failed!");
    }

    return F;
}

Function* KernelGenerator::emitLinear(const std::string& name) {
    Type* i16Ty     = Builder.getInt16Ty(); // BF16 is 16 bits (hint: It's in the name!)
    Type* floatTy   = Builder.getFloatTy();
    Type* i64Ty     = Builder.getInt64Ty();
    auto* vecTy     = FixedVectorType::get(floatTy, 8); // AVX 8-wide


    Builder.setFastMathFlags(FastMathFlags::getFast());

    std::vector<Type*> argTypes = {
        Builder.getPtrTy(), // Input
        Builder.getPtrTy(), // Weight
        Builder.getPtrTy(), // Output
        i64Ty,              // in_features
        i64Ty               // out_features
    };

    FunctionType* FT    = FunctionType::get(Builder.getVoidTy(), argTypes, false);
    Function* F         = Function::Create(FT, Function::ExternalLinkage, name, M);

    BasicBlock* Entry       = BasicBlock::Create(Ctx, "entry", F);
    BasicBlock* OutLoop     = BasicBlock::Create(Ctx, "out_loop", F);
    BasicBlock* InLoop      = BasicBlock::Create(Ctx, "in_loop", F);
    BasicBlock* Exit        = BasicBlock::Create(Ctx, "exit", F);
    BasicBlock* InLoopEnd   = BasicBlock::Create(Ctx, "in_loop_end", F);

    Builder.SetInsertPoint(Entry);
    Value* In           = F->getArg(0);
    Value* Weight       = F->getArg(1);
    Value* Out          = F->getArg(2);
    Value* InFeatures   = F->getArg(3);
    Value* OutFeatures  = F->getArg(4);

    Builder.CreateBr(OutLoop);

    Builder.SetInsertPoint(OutLoop);
    PHINode* RowIdx = Builder.CreatePHI(i64Ty, 2, "row_idx");
    RowIdx->addIncoming(Builder.getInt64(0), Entry);
    Builder.CreateBr(InLoop);

    // --- Inner Loop ---
    Builder.SetInsertPoint(InLoop);
    PHINode* ColIdx     = Builder.CreatePHI(i64Ty, 2, "col_idx"); // i64
    ColIdx->addIncoming(Builder.getInt64(0), OutLoop);

    PHINode* Accum      = Builder.CreatePHI(vecTy, 2, "accum");
    Accum->addIncoming(ConstantAggregateZero::get(vecTy), OutLoop);

    // 1. Load Input (FP32)
    Value* InPtr = Builder.CreateGEP(floatTy, In, ColIdx);
    Value* InVec = Builder.CreateLoad(vecTy, InPtr);

    // 2. Load Weight (BF16 -> FP32)
    Value* RowOff   = Builder.CreateMul(RowIdx, InFeatures);
    Value* WIdx     = Builder.CreateAdd(RowOff, ColIdx);

    // SCALE BY i16Ty (2 bytes)
    Value* WPtr     = Builder.CreateGEP(i16Ty, Weight, WIdx);
    Value* WVecI16  = Builder.CreateLoad(FixedVectorType::get(i16Ty, 8), WPtr);

    // BF16 to FP32 bit-shift trick
    Value* WVecI32      = Builder.CreateZExt(WVecI16, FixedVectorType::get(Builder.getInt32Ty(), 8));
    Value* WVecShifted  = Builder.CreateShl(WVecI32, ConstantVector::getSplat(ElementCount::getFixed(8), Builder.getInt32(16)));
    Value* WVec         = Builder.CreateBitCast(WVecShifted, vecTy);

    //math involved in mmult/dotprod or more accuratelt MAC (multiply and accumulate)
    Value* Mul          = Builder.CreateFMul(InVec, WVec);
    Value* NextAccum    = Builder.CreateFAdd(Accum, Mul);
    Value* NextCol      = Builder.CreateAdd(ColIdx, Builder.getInt64(8)); //matching i64 + i64

    ColIdx->addIncoming(NextCol, InLoop);
    Accum->addIncoming(NextAccum, InLoop);

    Builder.CreateCondBr(Builder.CreateICmpULT(NextCol, InFeatures), InLoop, InLoopEnd);

    //reduce
    Builder.SetInsertPoint(InLoopEnd);
    Value* ScalarSum    = Builder.CreateFAddReduce(ConstantFP::get(floatTy, 0.0f), NextAccum);
    Builder.CreateStore(ScalarSum, Builder.CreateGEP(floatTy, Out, RowIdx));

    Value* NextRow      = Builder.CreateAdd(RowIdx, Builder.getInt64(1));
    RowIdx->addIncoming(NextRow, InLoopEnd);
    Builder.CreateCondBr(Builder.CreateICmpULT(NextRow, OutFeatures), OutLoop, Exit);

    Builder.SetInsertPoint(Exit);
    Builder.CreateRetVoid();
    return F;
}

Function* KernelGenerator::emitSwiGLU(int size, const std::string& name) {
    Type* floatTy       = Builder.getFloatTy();
    Type* i64Ty         = Builder.getInt64Ty();
    auto* vecTy         = FixedVectorType::get(floatTy, 8);

    Builder.setFastMathFlags(FastMathFlags::getFast());

    
    FunctionType* FT    = FunctionType::get(Builder.getVoidTy(), 
        { Builder.getPtrTy(), Builder.getPtrTy(), Builder.getPtrTy() }, false);  // void swiglu(float* gate, float* value, float* output) <--- This is what we want.
    Function* F         = Function::Create(FT, Function::ExternalLinkage, name, M);

    BasicBlock* Entry   = BasicBlock::Create(Ctx, "entry", F);
    BasicBlock* Loop    = BasicBlock::Create(Ctx, "loop", F);
    BasicBlock* Exit    = BasicBlock::Create(Ctx, "exit", F);

    Builder.SetInsertPoint(Entry);
    Value* Gate         = F->getArg(0);
    Value* ValuePtr     = F->getArg(1);
    Value* Output       = F->getArg(2);

    Value* IVar = Builder.CreateAlloca(i64Ty, nullptr, "i_ptr");
    Builder.CreateStore(Builder.getInt64(0), IVar);
    Builder.CreateBr(Loop);

    Builder.SetInsertPoint(Loop);
    Value* I = Builder.CreateLoad(i64Ty, IVar);

    //load 8 floats from Gate and Value
    Value* GPtr = Builder.CreateGEP(floatTy, Gate, I);
    Value* GVec = Builder.CreateAlignedLoad(vecTy, GPtr, MaybeAlign(4));

    Value* VPtr = Builder.CreateGEP(floatTy, ValuePtr, I);
    Value* VVec = Builder.CreateAlignedLoad(vecTy, VPtr, MaybeAlign(4));

    //GeLU Approximation: 1.0 / (1.0 + exp(-1.702 * gate))
    Value* ScaleVec = ConstantVector::getSplat(ElementCount::getFixed(8), ConstantFP::get(floatTy, 1.702f));
    Value* ScaledG  = Builder.CreateFMul(GVec, ScaleVec);
    Value* NegG     = Builder.CreateFNeg(ScaledG);

    //intrinsic for vector of 8 floats
    Function* ExpFn = Intrinsic::getOrInsertDeclaration(M, Intrinsic::exp, { vecTy });
    Value* ExpVec   = Builder.CreateCall(ExpFn, { NegG });

    Value* OneVec   = ConstantVector::getSplat(ElementCount::getFixed(8), ConstantFP::get(floatTy, 1.0f));
    Value* Denom    = Builder.CreateFAdd(OneVec, ExpVec);
    Value* Sigmoid  = Builder.CreateFDiv(OneVec, Denom);  //<--- This can be done in a better way. Remeber, Division = "Not good".

    //gate * sigmoid(1.702 * gate)
    Value* Swish = Builder.CreateFMul(GVec, Sigmoid);

    //Swish * value
    Value* Final = Builder.CreateFMul(Swish, VVec);

    //store ts
    Value* OutPtr = Builder.CreateGEP(floatTy, Output, I);
    Builder.CreateAlignedStore(Final, OutPtr, MaybeAlign(4));

    //increment and Loop
    Value* NextI = Builder.CreateAdd(I, Builder.getInt64(8));
    Builder.CreateStore(NextI, IVar);

    Value* Cond = Builder.CreateICmpULT(NextI, Builder.getInt64(size));
    Builder.CreateCondBr(Cond, Loop, Exit);

    Builder.SetInsertPoint(Exit);
    Builder.CreateRetVoid();

    if (verifyFunction(*F, &errs())) throw std::runtime_error("SwiGLU Verification Failed");
    return F;
}

Function* KernelGenerator::emitRoPE(int head_dim, const std::string& name) {
    Type* floatTy = Builder.getFloatTy();
    Type* i64Ty = Builder.getInt64Ty();

    Builder.setFastMathFlags(FastMathFlags::getFast());

    FunctionType* FT = FunctionType::get(Builder.getVoidTy(),
        { Builder.getPtrTy(), Builder.getPtrTy(), Builder.getPtrTy() }, false);
    Function* F = Function::Create(FT, Function::ExternalLinkage, name, M);

    BasicBlock* Entry = BasicBlock::Create(Ctx, "entry", F);
    BasicBlock* Loop = BasicBlock::Create(Ctx, "loop", F);
    BasicBlock* Exit = BasicBlock::Create(Ctx, "exit", F);

    Builder.SetInsertPoint(Entry);
    Value* Vec = F->getArg(0);
    Value* CosTab = F->getArg(1);
    Value* SinTab = F->getArg(2);

    Value* IVar = Builder.CreateAlloca(i64Ty, nullptr, "i_ptr");
    Builder.CreateStore(Builder.getInt64(0), IVar);
    Builder.CreateBr(Loop);

    Builder.SetInsertPoint(Loop);
    Value* I = Builder.CreateLoad(i64Ty, IVar);

    //load the pair (x_i  and  x_{i + head_dim/2})
    Value* HalfDim  = Builder.getInt64(head_dim / 2);
    Value* PtrI     = Builder.CreateGEP(floatTy, Vec, I);
    Value* X_i      = Builder.CreateLoad(floatTy, PtrI);

    Value* NextI    = Builder.CreateAdd(I, HalfDim);
    Value* PtrNextI = Builder.CreateGEP(floatTy, Vec, NextI);
    Value* X_next   = Builder.CreateLoad(floatTy, PtrNextI);

    //load Cos and Sin for this dimension (from the precomputed tables)
    Value* C = Builder.CreateLoad(floatTy, Builder.CreateGEP(floatTy, CosTab, I));
    Value* S = Builder.CreateLoad(floatTy, Builder.CreateGEP(floatTy, SinTab, I));

    //apply Rotation (Half-and-Half format):
    Value* X_cos        = Builder.CreateFMul(X_i, C);
    Value* X_next_sin   = Builder.CreateFMul(X_next, S);
    Value* New_X        = Builder.CreateFSub(X_cos, X_next_sin);

    Value* X_sin        = Builder.CreateFMul(X_i, S);
    Value* X_next_cos   = Builder.CreateFMul(X_next, C);
    Value* New_X_next   = Builder.CreateFAdd(X_sin, X_next_cos);

    //store ts
    Builder.CreateStore(New_X, PtrI);
    Builder.CreateStore(New_X_next, PtrNextI);

    //increment by 1 (process up to head_dim / 2)
    Value* StepI    = Builder.CreateAdd(I, Builder.getInt64(1));
    Builder.CreateStore(StepI, IVar);

    Value* Cond     = Builder.CreateICmpULT(StepI, HalfDim);
    Builder.CreateCondBr(Cond, Loop, Exit);

    Builder.SetInsertPoint(Exit);
    Builder.CreateRetVoid();

    if (verifyFunction(*F, &errs())) throw std::runtime_error("RoPE Verification Failed");
    return F;
}

Function* KernelGenerator::emitSoftmax(const std::string& name) {
    Type* floatTy   = Builder.getFloatTy();
    Type* i64Ty     = Builder.getInt64Ty();

    Builder.setFastMathFlags(FastMathFlags::getFast());

    FunctionType* FT    = FunctionType::get(Builder.getVoidTy(), { Builder.getPtrTy(), i64Ty }, false); //void softmax(float* data, uint64_t size) <--- func signature looks like that.
    Function* F         = Function::Create(FT, Function::ExternalLinkage, name, M);

    //create blox
    BasicBlock* Entry       = BasicBlock::Create(Ctx, "entry", F);
    BasicBlock* InitBlock   = BasicBlock::Create(Ctx, "init_block", F);
    BasicBlock* MaxLoop     = BasicBlock::Create(Ctx, "max_loop", F);
    BasicBlock* ExpLoop     = BasicBlock::Create(Ctx, "exp_loop", F);
    BasicBlock* FinalLoop   = BasicBlock::Create(Ctx, "final_loop", F);
    BasicBlock* Exit        = BasicBlock::Create(Ctx, "exit", F);

    Builder.SetInsertPoint(Entry);
    Value* Data     = F->getArg(0);
    Value* SizeVal  = F->getArg(1); //retrieve size from caller

    //safety Guard: if (size == 0) { gtfo } ;
    Value* IsZero = Builder.CreateICmpEQ(SizeVal, Builder.getInt64(0));
    Builder.CreateCondBr(IsZero, Exit, InitBlock);

    Builder.SetInsertPoint(InitBlock);
    Value* IVar     = Builder.CreateAlloca(i64Ty, nullptr, "i_ptr");
    Value* MaxVar   = Builder.CreateAlloca(floatTy, nullptr, "max_ptr");
    Value* SumVar   = Builder.CreateAlloca(floatTy, nullptr, "sum_ptr");

    //initialize max with first element and I with 0
    Builder.CreateStore(Builder.CreateLoad(floatTy, Data), MaxVar);
    Builder.CreateStore(Builder.getInt64(0), IVar);
    Builder.CreateBr(MaxLoop);

    //the first step is to find the max
    Builder.SetInsertPoint(MaxLoop);
    Value* I1       = Builder.CreateLoad(i64Ty, IVar);
    Value* Val1     = Builder.CreateLoad(floatTy, Builder.CreateGEP(floatTy, Data, I1));
    Value* CurMax   = Builder.CreateLoad(floatTy, MaxVar);

    Value* NewMax   = Builder.CreateCall(Intrinsic::getOrInsertDeclaration(M, Intrinsic::maxnum, { floatTy }), { CurMax, Val1 }); //<--- There he is!
    Builder.CreateStore(NewMax, MaxVar);

    Value* NextI1   = Builder.CreateAdd(I1, Builder.getInt64(1));
    Builder.CreateStore(NextI1, IVar);
    //SizeVal instead of static size
    Builder.CreateCondBr(Builder.CreateICmpULT(NextI1, SizeVal), MaxLoop, ExpLoop);

    //Next is to accumulate the sum of exponents
    Builder.SetInsertPoint(ExpLoop);
    Builder.CreateStore(ConstantFP::get(floatTy, 0.0f), SumVar);
    Builder.CreateStore(Builder.getInt64(0), IVar);
    Value* FinalMax = Builder.CreateLoad(floatTy, MaxVar);

    BasicBlock* ExpBody = BasicBlock::Create(Ctx, "exp_body", F, FinalLoop);
    Builder.CreateBr(ExpBody);

    Builder.SetInsertPoint(ExpBody);
    Value* I2   = Builder.CreateLoad(i64Ty, IVar);
    Value* Val2 = Builder.CreateLoad(floatTy, Builder.CreateGEP(floatTy, Data, I2));

    Value* Diff     = Builder.CreateFSub(Val2, FinalMax);
    Value* ExpVal   = Builder.CreateCall(Intrinsic::getOrInsertDeclaration(M, Intrinsic::exp, { floatTy }), { Diff });

    Builder.CreateStore(ExpVal, Builder.CreateGEP(floatTy, Data, I2));
    Builder.CreateStore(Builder.CreateFAdd(Builder.CreateLoad(floatTy, SumVar), ExpVal), SumVar);

    Value* NextI2 = Builder.CreateAdd(I2, Builder.getInt64(1));
    Builder.CreateStore(NextI2, IVar);
    //use dynamic SizeVal instead of static size
    Builder.CreateCondBr(Builder.CreateICmpULT(NextI2, SizeVal), ExpBody, FinalLoop);

    //Next is to divide by the max exponent
    Builder.SetInsertPoint(FinalLoop);
    Value* FinalSum = Builder.CreateLoad(floatTy, SumVar);
    Builder.CreateStore(Builder.getInt64(0), IVar);

    BasicBlock* DivBody = BasicBlock::Create(Ctx, "div_body", F, Exit);
    Builder.CreateBr(DivBody);

    Builder.SetInsertPoint(DivBody);
    Value* I3 = Builder.CreateLoad(i64Ty, IVar);
    Value* Ptr3 = Builder.CreateGEP(floatTy, Data, I3);
    Value* CurrentExp = Builder.CreateLoad(floatTy, Ptr3);
    Builder.CreateStore(Builder.CreateFDiv(CurrentExp, FinalSum), Ptr3); //<--- This can be done in a better way. Remeber, Division = "Not good".

    Value* NextI3 = Builder.CreateAdd(I3, Builder.getInt64(1));
    Builder.CreateStore(NextI3, IVar);
    //SizeVal instead of static size
    Builder.CreateCondBr(Builder.CreateICmpULT(NextI3, SizeVal), DivBody, Exit);

    Builder.SetInsertPoint(Exit);
    Builder.CreateRetVoid();

    if (verifyFunction(*F, &errs())) throw std::runtime_error("Softmax Verification Failed");
    return F;
}

Function* KernelGenerator::emitKVCacheUpdate(int kv_dim, const std::string& name) {
    Type* floatTy   = Builder.getFloatTy();
    Type* i64Ty     = Builder.getInt64Ty();
    auto* vecTy     = FixedVectorType::get(floatTy, 8); // AVX 8-wide

    Builder.setFastMathFlags(FastMathFlags::getFast());

    FunctionType* FT = FunctionType::get(Builder.getVoidTy(), { Builder.getPtrTy(), Builder.getPtrTy() }, false); 
    Function* F = Function::Create(FT, Function::ExternalLinkage, name, M);

    BasicBlock* Entry = BasicBlock::Create(Ctx, "entry", F);
    BasicBlock* Loop = BasicBlock::Create(Ctx, "loop", F);
    BasicBlock* Exit = BasicBlock::Create(Ctx, "exit", F);

    Builder.SetInsertPoint(Entry);
    Value* NewKV = F->getArg(0);
    Value* CacheSlot = F->getArg(1);
    Value* IVar = Builder.CreateAlloca(i64Ty, nullptr, "i_ptr");
    Builder.CreateStore(Builder.getInt64(0), IVar);
    Builder.CreateBr(Loop);

    Builder.SetInsertPoint(Loop);
    Value* I = Builder.CreateLoad(i64Ty, IVar);

    //copy as vector
    Value* SrcPtr = Builder.CreateGEP(floatTy, NewKV, I);
    Value* DstPtr = Builder.CreateGEP(floatTy, CacheSlot, I);

    Value* Vec = Builder.CreateAlignedLoad(vecTy, SrcPtr, MaybeAlign(4));
    Builder.CreateAlignedStore(Vec, DstPtr, MaybeAlign(4));

    Value* NextI = Builder.CreateAdd(I, Builder.getInt64(8));
    Builder.CreateStore(NextI, IVar);
    Builder.CreateCondBr(Builder.CreateICmpULT(NextI, Builder.getInt64(kv_dim)), Loop, Exit);

    Builder.SetInsertPoint(Exit);
    Builder.CreateRetVoid();

    return F;
}

Function* KernelGenerator::emitAttentionScore(int head_dim, const std::string& name) {
    Type* floatTy = Builder.getFloatTy();
    Type* i64Ty = Builder.getInt64Ty();
    auto* vecTy = FixedVectorType::get(floatTy, 8);

    Builder.setFastMathFlags(FastMathFlags::getFast());

    FunctionType* FT = FunctionType::get(Builder.getVoidTy(),
        { Builder.getPtrTy(), Builder.getPtrTy(), Builder.getPtrTy(), i64Ty }, false);
    Function* F = Function::Create(FT, Function::ExternalLinkage, name, M);

    BasicBlock* Entry = BasicBlock::Create(Ctx, "entry", F);
    BasicBlock* PosLoop = BasicBlock::Create(Ctx, "pos_loop", F);
    BasicBlock* InLoop = BasicBlock::Create(Ctx, "in_loop", F);
    BasicBlock* NextPos = BasicBlock::Create(Ctx, "next_pos", F);
    BasicBlock* Exit = BasicBlock::Create(Ctx, "exit", F);

    Builder.SetInsertPoint(Entry);
    Value* Q = F->getArg(0);
    Value* KCache = F->getArg(1);
    Value* Scores = F->getArg(2);
    Value* CurrentPos = F->getArg(3);

    Value* PVar = Builder.CreateAlloca(i64Ty, nullptr, "p_ptr");
    Builder.CreateStore(Builder.getInt64(0), PVar);
    Builder.CreateBr(PosLoop);

    Builder.SetInsertPoint(PosLoop);
    Value* P = Builder.CreateLoad(i64Ty, PVar);

    //ONLY prefetch if there's a next position to fetch
    Value* NextP_ForPrefetch = Builder.CreateAdd(P, Builder.getInt64(1));
    Value* HasNext = Builder.CreateICmpULE(NextP_ForPrefetch, CurrentPos);

    //Create a temporary block for prefetch to keep IR clean
    BasicBlock* PrefetchBB = BasicBlock::Create(Ctx, "prefetch_block", F, InLoop);
    BasicBlock* SkipPrefetchBB = BasicBlock::Create(Ctx, "skip_prefetch", F, InLoop);
    Builder.CreateCondBr(HasNext, PrefetchBB, SkipPrefetchBB);

    Builder.SetInsertPoint(PrefetchBB);
    Value* PrefetchKOff = Builder.CreateMul(NextP_ForPrefetch, Builder.getInt64(head_dim));
    Value* PrefetchPtr = Builder.CreateGEP(floatTy, KCache, PrefetchKOff);
    Function* PrefetchFn = Intrinsic::getOrInsertDeclaration(M, Intrinsic::prefetch, { Builder.getPtrTy() });
    Builder.CreateCall(PrefetchFn, { PrefetchPtr, Builder.getInt32(0), Builder.getInt32(3), Builder.getInt32(1) });
    Builder.CreateBr(SkipPrefetchBB);

    Builder.SetInsertPoint(SkipPrefetchBB);
    //accumulator and inner index reset.
    Value* AccumVar = Builder.CreateAlloca(vecTy, nullptr, "accum_ptr");
    Builder.CreateStore(ConstantAggregateZero::get(vecTy), AccumVar);
    Value* IVar = Builder.CreateAlloca(i64Ty, nullptr, "i_ptr");
    Builder.CreateStore(Builder.getInt64(0), IVar);
    Builder.CreateBr(InLoop);

    //SIMD Dot Product (Q * K_p)
    Builder.SetInsertPoint(InLoop);
    Value* I = Builder.CreateLoad(i64Ty, IVar);

    Value* QVec = Builder.CreateAlignedLoad(vecTy, Builder.CreateGEP(floatTy, Q, I), MaybeAlign(4));
    Value* KOff = Builder.CreateAdd(Builder.CreateMul(P, Builder.getInt64(head_dim)), I);
    Value* KVec = Builder.CreateAlignedLoad(vecTy, Builder.CreateGEP(floatTy, KCache, KOff), MaybeAlign(4));

    Value* NextAccum = Builder.CreateFAdd(Builder.CreateLoad(vecTy, AccumVar), Builder.CreateFMul(QVec, KVec));
    Builder.CreateStore(NextAccum, AccumVar);

    Value* NextI = Builder.CreateAdd(I, Builder.getInt64(8));
    Builder.CreateStore(NextI, IVar);
    Builder.CreateCondBr(Builder.CreateICmpULT(NextI, Builder.getInt64(head_dim)), InLoop, NextPos);

    //finalize the score for position p, then we're basically done.
    Builder.SetInsertPoint(NextPos);
    Value* Sum = Builder.CreateFAddReduce(ConstantFP::get(floatTy, 0.0f), Builder.CreateLoad(vecTy, AccumVar));
    float scale = 1.0f / std::sqrt((float)head_dim);
    Value* ScaledSum = Builder.CreateFMul(Sum, ConstantFP::get(floatTy, scale));
    Builder.CreateAlignedStore(ScaledSum, Builder.CreateGEP(floatTy, Scores, P), MaybeAlign(4));

    Value* NextP = Builder.CreateAdd(P, Builder.getInt64(1));
    Builder.CreateStore(NextP, PVar);
    Builder.CreateCondBr(Builder.CreateICmpULE(NextP, CurrentPos), PosLoop, Exit);

    Builder.SetInsertPoint(Exit);
    Builder.CreateRetVoid();
    return F;
}

Function* KernelGenerator::emitAttentionValueSum(int head_dim, const std::string& name) {
    Type* floatTy = Builder.getFloatTy();
    Type* i64Ty = Builder.getInt64Ty();
    auto* vecTy = FixedVectorType::get(floatTy, 8);

    Builder.setFastMathFlags(FastMathFlags::getFast());

    FunctionType* FT = FunctionType::get(Builder.getVoidTy(), 
        { Builder.getPtrTy(), Builder.getPtrTy(), Builder.getPtrTy(), i64Ty }, false); // void attention_value_sum(float* scores, float* v_cache, float* output, int current_pos) <--- Function signature should look like this.
    Function* F = Function::Create(FT, Function::ExternalLinkage, name, M);

    BasicBlock* Entry = BasicBlock::Create(Ctx, "entry", F);
    BasicBlock* DimLoop = BasicBlock::Create(Ctx, "dim_loop", F);
    BasicBlock* PosLoop = BasicBlock::Create(Ctx, "pos_loop", F);
    BasicBlock* NextDim = BasicBlock::Create(Ctx, "next_dim", F);
    BasicBlock* Exit = BasicBlock::Create(Ctx, "exit", F);

    Builder.SetInsertPoint(Entry);
    Value* Scores = F->getArg(0);
    Value* VCache = F->getArg(1);
    Value* Output = F->getArg(2);
    Value* CurrentPos = F->getArg(3);

    Value* DVar = Builder.CreateAlloca(i64Ty, nullptr, "d_ptr"); // Dimension index
    Builder.CreateStore(Builder.getInt64(0), DVar);
    Builder.CreateBr(DimLoop);

    // --- Outer Loop: Iterate over Head Dimension (8 at a time) ---
    Builder.SetInsertPoint(DimLoop);
    Value* D = Builder.CreateLoad(i64Ty, DVar);

    Value* AccumVar = Builder.CreateAlloca(vecTy, nullptr, "accum_ptr");
    Builder.CreateStore(ConstantAggregateZero::get(vecTy), AccumVar);

    Value* PVar = Builder.CreateAlloca(i64Ty, nullptr, "p_ptr"); // Position index
    Builder.CreateStore(Builder.getInt64(0), PVar);
    Builder.CreateBr(PosLoop);

    // --- Inner Loop: Accumulate Weighted Values across positions ---
    Builder.SetInsertPoint(PosLoop);
    Value* P = Builder.CreateLoad(i64Ty, PVar);

    // 1. Load the scalar score and splat it
    Value* ScoreVal = Builder.CreateLoad(floatTy, Builder.CreateGEP(floatTy, Scores, P));
    Value* ScoreSplat = Builder.CreateVectorSplat(8, ScoreVal);

    // 2. Load the Value vector: V_cache[P * head_dim + D]
    Value* VOff = Builder.CreateAdd(Builder.CreateMul(P, Builder.getInt64(head_dim)), D);
    Value* VVec = Builder.CreateAlignedLoad(vecTy, Builder.CreateGEP(floatTy, VCache, VOff), MaybeAlign(4));

    // 3. FMA: accum = (VVec * ScoreSplat) + accum
    Value* WeightedV = Builder.CreateFMul(VVec, ScoreSplat);
    Value* NewAccum = Builder.CreateFAdd(Builder.CreateLoad(vecTy, AccumVar), WeightedV);
    Builder.CreateStore(NewAccum, AccumVar);

    Value* NextP = Builder.CreateAdd(P, Builder.getInt64(1));
    Builder.CreateStore(NextP, PVar);
    Builder.CreateCondBr(Builder.CreateICmpULE(NextP, CurrentPos), PosLoop, NextDim);

    // --- Finalize: Store resulting vector to Output ---
    Builder.SetInsertPoint(NextDim);
    Value* FinalVec = Builder.CreateLoad(vecTy, AccumVar);
    Builder.CreateAlignedStore(FinalVec, Builder.CreateGEP(floatTy, Output, D), MaybeAlign(4));

    Value* NextD = Builder.CreateAdd(D, Builder.getInt64(8));
    Builder.CreateStore(NextD, DVar);
    Builder.CreateCondBr(Builder.CreateICmpULT(NextD, Builder.getInt64(head_dim)), DimLoop, Exit);

    Builder.SetInsertPoint(Exit);
    Builder.CreateRetVoid();

    if (verifyFunction(*F, &errs())) throw std::runtime_error("Weighted Sum Verification Failed");
    return F;
}

Function* KernelGenerator::emitEmbeddingLookup(int hidden_size, const std::string& name) {
    Type* i16Ty = Builder.getInt16Ty();
    Type* i64Ty = Builder.getInt64Ty();
    Type* floatTy = Builder.getFloatTy();
    auto* vecTy = FixedVectorType::get(floatTy, 8);

    Builder.setFastMathFlags(FastMathFlags::getFast());

    FunctionType* FT = FunctionType::get(Builder.getVoidTy(),
        { i64Ty, Builder.getPtrTy(), Builder.getPtrTy() }, false);
    Function* F = Function::Create(FT, Function::ExternalLinkage, name, M);

    BasicBlock* Entry = BasicBlock::Create(Ctx, "entry", F);
    BasicBlock* Loop = BasicBlock::Create(Ctx, "loop", F);
    BasicBlock* Exit = BasicBlock::Create(Ctx, "exit", F);

    Builder.SetInsertPoint(Entry);
    Value* TokenID = F->getArg(0);
    Value* Table = F->getArg(1);
    Value* Output = F->getArg(2);

    float scale_val = std::sqrt((float)hidden_size);
    Value* ScaleVec = Builder.CreateVectorSplat(8, ConstantFP::get(floatTy, scale_val));
    Value* RowOffset = Builder.CreateMul(TokenID, Builder.getInt64(hidden_size));

    Builder.CreateBr(Loop);

    // --- Loop: Using PHI Nodes (No Stack/Alloca) ---
    Builder.SetInsertPoint(Loop);
    PHINode* I = Builder.CreatePHI(i64Ty, 2, "i");
    I->addIncoming(Builder.getInt64(0), Entry);

    // 1. Load from table (BF16 -> 2-byte scaling)
    Value* TableIdx = Builder.CreateAdd(RowOffset, I);
    Value* TablePtr = Builder.CreateGEP(i16Ty, Table, TableIdx); // FIXED: i16Ty
    Value* VecI16 = Builder.CreateLoad(FixedVectorType::get(i16Ty, 8), TablePtr);

    // 2. Convert BF16 to FP32
    Value* VecI32 = Builder.CreateZExt(VecI16, FixedVectorType::get(Builder.getInt32Ty(), 8));
    Value* VecShifted = Builder.CreateShl(VecI32, ConstantVector::getSplat(ElementCount::getFixed(8), Builder.getInt32(16)));
    Value* Vec = Builder.CreateBitCast(VecShifted, vecTy);

    // 3. Scale and Store to h_state
    Value* ScaledVec = Builder.CreateFMul(Vec, ScaleVec);
    //Value* ScaledVec = Vec;
    Value* OutPtr = Builder.CreateGEP(floatTy, Output, I);
    Builder.CreateAlignedStore(ScaledVec, OutPtr, MaybeAlign(32)); // Requires 32-byte aligned h_state

    Value* NextI = Builder.CreateAdd(I, Builder.getInt64(8));
    I->addIncoming(NextI, Loop);

    Builder.CreateCondBr(Builder.CreateICmpULT(NextI, Builder.getInt64(hidden_size)), Loop, Exit);

    Builder.SetInsertPoint(Exit);
    Builder.CreateRetVoid();
    return F;
}