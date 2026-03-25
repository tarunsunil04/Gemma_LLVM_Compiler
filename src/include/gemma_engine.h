#pragma once
#include "gemma_tokenizer.h"
#include "config.h"
#include "jit.h"
#include "kernel_generator.h"
#include "kv_cache.h"
#include "model_loader.h"

#include <string>
#include <vector>
#include <memory>

template <typename T>
struct AlignedDeleter {
    void operator()(T* p) const { _aligned_free(p); }
};

template <typename T>
using AlignedPtr = std::unique_ptr<T[], AlignedDeleter<T>>;

template <typename T>
AlignedPtr<T> make_aligned(size_t size) {
    auto ptr = AlignedPtr<T>((T*)_aligned_malloc(size * sizeof(T), 32));
    std::memset(ptr.get(), 0, size * sizeof(T));
    return ptr;
}

typedef void (*RMSNormFn)(float*, float*, float*);
typedef void (*LinearFn)(float* , void* , float* , uint64_t, uint64_t);
typedef void (*SwiGLUFn)(float*, float*, float*);
typedef void (*RoPEFn)(float*, float*, float*, uint64_t);
typedef void (*KVUpdateFn)(float*, float*);
typedef void (*SoftmaxFn)(float*, uint64_t);
typedef void (*ScoreFn)(float*, float*, float*, uint64_t);
typedef void (*SumFn)(float*, float*, float*, uint64_t);
typedef void (*EmbedFn)(uint64_t, void*, float*);

struct LayerWeights {
    float* input_norm;
    float* q_proj;
    float* k_proj;
    float* v_proj;
    float* o_proj;
    float* post_norm;
    float* gate_proj;
    float* up_proj;
    float* down_proj;
};


class GemmaEngine {
public:
    GemmaEngine(const std::vector<std::string>& shard_paths, const std::string& tokenizer_path);

    std::string generate_from_tokens(const std::vector<int>& input_tokens, int max_tokens);
    std::string generate(const std::string& prompt, int max_tokens);
    GemmaTokenizer  tokenizer;


private:  
    GemmaConfig     config;
    ModelLoader     loader;
    GemmaJIT        jit;
    KVCache         kv_cache;
    


    // Global weights
    float* embed_tokens_w   = nullptr;
    float* final_norm_w     = nullptr;

    RMSNormFn       rms_norm_ptr = nullptr;
    RoPEFn          rope_ptr     = nullptr;
    ScoreFn         score_ptr    = nullptr;
    SwiGLUFn        swiglu_ptr   = nullptr;
    KVUpdateFn      kv_up_ptr    = nullptr;
    SoftmaxFn       softmax_ptr  = nullptr;
    SumFn           sum_ptr      = nullptr;
    EmbedFn         embed_ptr    = nullptr;

    LinearFn        linear_ptr          = nullptr; // 2048 -> 2048
    LinearFn        linear_up_ptr       = nullptr; // 2048 -> 16384
    LinearFn        linear_down_ptr     = nullptr; // 16384 -> 2048
    LinearFn        linear_vocab_ptr    = nullptr; // 2048 -> 256000 

    AlignedPtr<float> h_state, temp_buf, q_buf, k_buf, v_buf;                               //<--|__These are temp buffers to store intermidiate results of each layer.
    AlignedPtr<float> gate_buf, up_buf, att_out, logits, cos_table, sin_table, scores_buf;  //<--|

    std::vector<LayerWeights> layer_data;

    //Internal methods
    int sample_argmax();
    int sample_top_k(float temperature = 0.8f, int top_k = 40);
    int forward(int token_id, int pos);
    void parallel_linear(LinearFn kernel, float* in, float* weight, float* out, uint64_t in_features, uint64_t out_features);
    float* get_weight(const std::string& name);
    void audit_system_health(int layer, const std::string& weight_name); //<-- Not used anymore, but it checks the weights referenced and its values are valid or not, during a forward pass of the model.
};