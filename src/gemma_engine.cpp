#include "include/gemma_engine.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include "include/kernel_generator.h"

#include <omp.h>
#include <intrin.h>

#include <random>
#include <vector>
#include <numeric>
#include <chrono>

#include <memory>
#include <malloc.h>

#include <windows.h>
#include <psapi.h>

//========================================DEBUG==============================================

void print_ram_usage() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
    std::cout << "[RAM Usage] " << (pmc.WorkingSetSize / 1024 / 1024) << " MB" << std::endl;
}

void GemmaEngine::audit_system_health(int layer, const std::string& weight_name) {
    void* stack_ptr = _AddressOfReturnAddress();
    float* w_ptr = get_weight(weight_name);

    std::cout << "\n[Layer " << layer << " Health Check]" << std::endl;
    std::cout << "  - Stack Address: " << stack_ptr << std::endl;
    std::cout << "  - Weight Target: " << weight_name << std::endl;
    std::cout << "  - Weight Pointer: " << (w_ptr ? "VALID" : "NULL/INVALID") << std::endl;

    if (!w_ptr) {
        std::cerr << "!!! MEMORY SCREWED: ModelLoader failed at Layer " << layer << std::endl;
        throw std::runtime_error("Null weight detected.");
    }
}
//===========================================================================================

GemmaEngine::GemmaEngine(const std::vector<std::string>& shard_paths,
    const std::string& tokenizer_path)
    : loader(shard_paths),
    kv_cache(config, 2048),
    tokenizer(tokenizer_path) {

    h_state =       make_aligned<float>(config.hidden_size);
    temp_buf =      make_aligned<float>(config.hidden_size);
    q_buf =         make_aligned<float>(config.hidden_size);
    k_buf =         make_aligned<float>(config.head_dim);
    v_buf =         make_aligned<float>(config.head_dim);
    gate_buf =      make_aligned<float>(config.intermediate_size);
    up_buf =        make_aligned<float>(config.intermediate_size);
    logits =        make_aligned<float>(256000);
    scores_buf =    make_aligned<float>(config.num_heads * 2048);
    att_out =       make_aligned<float>(config.hidden_size);


    //The RoPE values are already computed beforehand.
    cos_table = make_aligned<float>(2048 * config.head_dim);
    sin_table = make_aligned<float>(2048 * config.head_dim);

    for (int pos = 0; pos < 2048; ++pos) 
    {
        for (int i = 0; i < config.head_dim / 2; ++i) 
        {
            float freq = 1.0f / std::pow(10000.0f, (float)(2 * i) / config.head_dim);
            float val = pos * freq;

            int idx = pos * config.head_dim + i;

            cos_table[idx] = std::cos(val);
            sin_table[idx] = std::sin(val);
        }
    }

    embed_tokens_w = get_weight("model.embed_tokens.weight");
    final_norm_w = get_weight("model.norm.weight");

    if (!embed_tokens_w || !final_norm_w) { throw std::runtime_error("Fatal: Global weight mapping failed."); }

    //Resolve Layer Weights by their id's. A safetensor file is basically a giant JSON that stores layer information in this format.  
    for (int i = 0; i < config.num_layers; ++i) {
        std::string p = "model.layers." + std::to_string(i);
        LayerWeights lw;
        lw.input_norm = get_weight(p + ".input_layernorm.weight");
        lw.q_proj =     get_weight(p + ".self_attn.q_proj.weight");
        lw.k_proj =     get_weight(p + ".self_attn.k_proj.weight");
        lw.v_proj =     get_weight(p + ".self_attn.v_proj.weight");
        lw.o_proj =     get_weight(p + ".self_attn.o_proj.weight");
        lw.post_norm =  get_weight(p + ".post_attention_layernorm.weight");
        lw.gate_proj =  get_weight(p + ".mlp.gate_proj.weight");
        lw.up_proj =    get_weight(p + ".mlp.up_proj.weight");
        lw.down_proj =  get_weight(p + ".mlp.down_proj.weight");

        if (!lw.input_norm || !lw.gate_proj) {
            throw std::runtime_error("Weight resolution failed at Layer " + std::to_string(i)); //<-- Check to see if the weight pointers are consistent
        }
        layer_data.push_back(lw);
    }

    //Cold starting the JIT engine, 1.5TDI style.
    auto M = std::make_unique<llvm::Module>("GemmaKernels", jit.getGlobalContext());
    KernelGenerator gen(jit.getGlobalContext(), M.get(), config);

    std::cout << "[GemmaEngine] JITing Kernels..." << std::endl;
    gen.emitEmbeddingLookup(config.hidden_size, "gemma_embedding_lookup");
    gen.emitRMSNorm();
    gen.emitLinear("gemma_linear_layer");
    gen.emitLinear("gemma_linear_up");
    gen.emitLinear("gemma_linear_down");
    gen.emitLinear("gemma_linear_vocab");
    gen.emitSwiGLU(config.intermediate_size, "gemma_swiglu");
    gen.emitRoPE(config.head_dim, "gemma_rope");
    gen.emitKVCacheUpdate(config.head_dim, "update_kv_kernel");
    gen.emitAttentionScore(config.head_dim, "gemma_attention_score");
    gen.emitAttentionValueSum(config.head_dim, "gemma_attention_sum");
    gen.emitSoftmax("gemma_softmax");

    //setting up the optimization params to make use of LLVM's in-built optimization capabilities
    jit.optimizeModule(*M);     //<---This allows us to use O2 optimization via the function pass manager that we defined in jit_engine.
    jit.addModule(std::move(M), std::make_unique<llvm::LLVMContext>());


    //assigning the pointers to all the functions that we'll be using in the layers, that'll be important later. Trsust me on this.
    embed_ptr =         (EmbedFn)jit.getFunctionPtr("gemma_embedding_lookup");
    rms_norm_ptr =      (RMSNormFn)jit.getFunctionPtr("gemma_rmsnorm");
    swiglu_ptr =        (SwiGLUFn)jit.getFunctionPtr("gemma_swiglu");
    rope_ptr =          (RoPEFn)jit.getFunctionPtr("gemma_rope");
    kv_up_ptr =         (KVUpdateFn)jit.getFunctionPtr("update_kv_kernel");
    score_ptr =         (ScoreFn)jit.getFunctionPtr("gemma_attention_score");
    softmax_ptr =       (SoftmaxFn)jit.getFunctionPtr("gemma_softmax");
    sum_ptr =           (SumFn)jit.getFunctionPtr("gemma_attention_sum");
    linear_ptr =        (LinearFn)jit.getFunctionPtr("gemma_linear_layer");
    linear_up_ptr =     (LinearFn)jit.getFunctionPtr("gemma_linear_up");
    linear_down_ptr =   (LinearFn)jit.getFunctionPtr("gemma_linear_down");
    linear_vocab_ptr =  (LinearFn)jit.getFunctionPtr("gemma_linear_vocab");

    if (!get_weight("model.embed_tokens.weight")) {
        throw std::runtime_error("Fatal: model.embed_tokens.weight missing from shards.");
    }

    std::cout << "[GemmaEngine] Initialization complete. Engine ready." << std::endl;
}


/* This function below is the chief reason for the speed that we get. If we were to use the non-parallelized (I'm very sure thats not a word) version, the TTFT anf TPS 
measures would be double and half respectively of what we have now. By assigning a set of available threads to do the task of the LinearFn (mmult), we've effectively used
the infrastructure that we've created, to its fullest extent.*/ 

void GemmaEngine::parallel_linear(LinearFn kernel, float* in, float* weight, float* out, uint64_t in_features, uint64_t out_features) {
    int num_threads = omp_get_max_threads();
    uint64_t chunk_size = out_features / num_threads;

#pragma omp parallel for
    for (int t = 0; t < num_threads; ++t) {
        uint64_t start_row = t * chunk_size;

        // Handle remainder for the last thread if out_features isn't perfectly divisible
        uint64_t rows_for_this_thread = (t == num_threads - 1) ? (out_features - start_row) : chunk_size;

        // Pointer arithmetic for BF16 (2 bytes per weight)
        float* weight_offset = (float*)((int16_t*)weight + (start_row * in_features));
        float* out_offset = out + start_row;

        // Dispatch to the specifically optimized JIT kernel
        kernel(in, weight_offset, out_offset, in_features, rows_for_this_thread);
    }
}


/*I spent an embrassingly long time to debug the forward loop, and from what I could learn, clear labelling and proper understanding of the forward loop is really 
what matters ultimately. Writing this in CPP doesn't do any favours, as if there's anything wrong, all you get is an access violation (stack overflow). So this is quite important.*/

int GemmaEngine::forward(int token_id, int pos) {
    float* h = h_state.get();
    float* t = temp_buf.get();

    //Initial Embedding
    embed_ptr((uint64_t)token_id, embed_tokens_w, h);

    //Main Transformer Loop (Should run for 18 layers, in one forward pass. This is quite useful information while debugging this method. Believe me.) 
    for (int l = 0; l < config.num_layers; ++l) {
        const auto& w = layer_data[l];

        rms_norm_ptr(h, w.input_norm, t);  //<-- This is where the attention block starts

        //Q, K, V Projections
        
        parallel_linear(linear_ptr, t, w.q_proj, q_buf.get(), 2048ULL, 2048ULL);
        parallel_linear(linear_ptr, t, w.k_proj, k_buf.get(), 2048ULL, 256ULL);
        parallel_linear(linear_ptr, t, w.v_proj, v_buf.get(), 2048ULL, 256ULL);

        //RoPE init
        float* c = cos_table.get() + (pos * config.head_dim);
        float* s = sin_table.get() + (pos * config.head_dim);

        //apply RoPE to 8 independent Query heads
        for (int head = 0; head < config.num_heads; ++head) {
            rope_ptr(q_buf.get() + (head * config.head_dim), c, s, (uint64_t)config.head_dim);
        }
        //apply RoPE to 1 shared Key head 
        rope_ptr(k_buf.get(), c, s, (uint64_t)config.head_dim);

        
        //we update the KV cache over here. 
        kv_up_ptr(k_buf.get(), kv_cache.get_layer_kv(l, false, pos));
        kv_up_ptr(v_buf.get(), kv_cache.get_layer_kv(l, true, pos));

        //MQA starts here:
        for (int head = 0; head < config.num_heads; ++head) {
            //calculate pointer offsets for the current head
            float* q_head = q_buf.get() + (head * config.head_dim);
            float* score_head = scores_buf.get() + (head * 2048); //requires scores_buf to be sized (8 * 2048)
            float* out_head = att_out.get() + (head * config.head_dim);

            //Calculate scores, apply dynamic softmax, and sum values
            score_ptr(q_head, kv_cache.get_layer_kv(l, false, 0), score_head, (uint64_t)pos);
            softmax_ptr(score_head, (uint64_t)(pos + 1));
            sum_ptr(score_head, kv_cache.get_layer_kv(l, true, 0), out_head, (uint64_t)pos);
        }

        //Output Projection & Residual 1
        parallel_linear(linear_ptr, att_out.get(), w.o_proj, t, 2048ULL, 2048ULL);
        for (int i = 0; i < 2048; ++i) h[i] += t[i];

        //MLP SEC
        rms_norm_ptr(h, w.post_norm, t);

        //Gate and Up Projections
        parallel_linear(linear_ptr, t, w.gate_proj, gate_buf.get(), 2048ULL, 16384ULL);
        parallel_linear(linear_ptr, t, w.up_proj, up_buf.get(), 2048ULL, 16384ULL);

        //paassing through the Activation Function
        swiglu_ptr(gate_buf.get(), up_buf.get(), gate_buf.get());

        //down Projection & Residual 2
        parallel_linear(linear_ptr, gate_buf.get(), w.down_proj, t, 16384ULL, 2048ULL);
        for (int i = 0; i < 2048; ++i) h[i] += t[i];
    }

    //Final Norm && Vocabulary Projection
    rms_norm_ptr(h, final_norm_w, h);
    linear_vocab_ptr(h, embed_tokens_w, logits.get(), 2048ULL, 256000ULL);

    return sample_top_k(0.8f, 40);
}

int GemmaEngine::sample_argmax() {
    int max_idx = 0;
    float max_val = -1e30f;
    for (int i = 0; i < 256000; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

/* If you stuck around, you know that this function really orchestrates everything. The timing mechanisms aren't really that necessary, but they're useful for debugging and 
observation purposes. */

std::string GemmaEngine::generate(const std::string& prompt, int max_tokens) {

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<int> tokens = tokenizer.encode(prompt);
    if (tokens.empty()) return "";

    if (tokens[0] != 2) {
        tokens.insert(tokens.begin(), 2);
    }

    std::string result = "";
    int next_token = 0;
    int max_context_window = 2048; //<--- This is tied directly to the KV cacge allocation limit.

    //prefill Phase: Process all prompt tokens except the last one (this is the case for most generate loops)
    for (size_t i = 0; i < tokens.size() - 1; ++i) {
        if (i >= max_context_window) {
            std::cerr << "\n[Error] Prompt exceeds maximum context window of 2048." << std::endl;
            return "";
        }
        forward(tokens[i], (int)i);
    }

    //forward the very last prompt token
    int current_pos = (int)tokens.size() - 1;
    if (current_pos >= max_context_window) return "";

    next_token = forward(tokens.back(), current_pos);
    auto first_token_time = std::chrono::high_resolution_clock::now();

    double ttft = std::chrono::duration<double, std::milli>(first_token_time - start_time).count();
    std::cout << "\n[Metrics] TTFT: " << ttft << " ms" << std::endl;

    int generated_count = 0;
    auto decode_start = std::chrono::high_resolution_clock::now();

    //decode Phase (Autoregressive Loop)
    for (int step = current_pos + 1; step < current_pos + max_tokens; ++step) {
        //prevent Buffer Overflow in JIT Kernels
        if (step >= max_context_window) {
            std::cout << "\n[Warning] Context window limit (2048) reached. Stopping generation." << std::endl;
            break;
        }

        //decode the integer back to text and stream to console smoothly
        std::string part = tokenizer.decode(next_token);
        result += part;
        std::cout << part << std::flush;


        generated_count++;
        //stop conditions: <eos> (1) or <pad> (0)   //<---Some resources say that token id 107 also works as an <eos>, but it seems to break the entire thing when I add that to this condition. 
        if (next_token == 1 || next_token == 0) {
            break;
        }

        //Feed the predicted token back in to get the next one
        next_token = forward(next_token, step);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_decode_time = std::chrono::duration<double>(end_time - decode_start).count();
    std::cout << "\n[Metrics] Speed: " << (generated_count / total_decode_time) << " tokens/sec" << std::endl;

    print_ram_usage();


    std::cout << std::endl;
    return result;
}

float* GemmaEngine::get_weight(const std::string& name) {
    try {
        auto tensor = loader.get_tensor(name);
        return (float*)tensor.data;
    }
    catch (...) {
        return nullptr;
    }
}

/* The decoding strategy over here is something similar to top-k with temperature, and does it's task accordingly. I found this resource that explain the different 
decoding strategies: https://medium.com/google-cloud/sampling-in-llms-14b213b6d704 
I also found this to be quite helpful: https://github.com/kmkarakaya/Deep-Learning-Tutorials/blob/master/Sampling_in_Text_Generation.ipynb */

int GemmaEngine::sample_top_k(float temperature, int top_k) {
    int vocab_size = 256000;

    static std::vector<int> history;
    for (int id : history) {
        
        if (logits[id] > 0) logits[id] /= 1.2f; //<--- We're essentially trying to simulate punishing the model for repeating words again and again, via temperature.
        else logits[id] *= 1.2f; //<--- neg values
    }

    //If temp is 0, fallback to greedy decoding. 
    if (temperature < 1e-5f) {
        int max_idx = 0;
        float max_val = -1e30f;
        for (int i = 0; i < vocab_size; ++i) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        return max_idx;
    }

    //Pair logits with their corresponding Token IDs
    std::vector<std::pair<float, int>> token_probs;
    token_probs.reserve(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        //Apply Temperature scaling
        token_probs.push_back({ logits[i] / temperature, i });
    }

    //Partial Sort to isolate the Top-K tokens
    top_k = std::min(top_k, vocab_size);
    std::partial_sort(token_probs.begin(), token_probs.begin() + top_k, token_probs.end(),
        [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first > b.first;
        });

    //Numerically Stable Softmax over Top-K
    float max_logit = token_probs[0].first;
    float sum_probs = 0.0f;
    std::vector<float> probs(top_k);

    for (int i = 0; i < top_k; ++i) {
        probs[i] = std::exp(token_probs[i].first - max_logit);
        sum_probs += probs[i];
    }

    //Roulette Wheel Selection (Sampling)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, sum_probs);
    float r = dis(gen);

    float accum = 0.0f;
    for (int i = 0; i < top_k; ++i) {
        accum += probs[i];
        if (r <= accum) {
            return token_probs[i].second;
        }
    }

    int chosen_token = token_probs[top_k - 1].second; //Or wherever you return

    history.push_back(chosen_token);
    if (history.size() > 50) history.erase(history.begin()); //Remember last 50 tokens

    return chosen_token;
}

std::string GemmaEngine::generate_from_tokens(const std::vector<int>& input_tokens, int max_tokens) {
    auto start_time = std::chrono::high_resolution_clock::now();

    if (input_tokens.empty()) return "";

    std::string result = "";
    int next_token = 0;
    int max_context_window = 2048; // Tied directly to the KV cache allocation limit.

    // --- PREFILL PHASE ---
    for (size_t i = 0; i < input_tokens.size() - 1; ++i) {
        if (i >= max_context_window) {
            std::cerr << "\n[Error] Prompt exceeds maximum context window of 2048." << std::endl;
            return "";
        }
        forward(input_tokens[i], (int)i);
    }

    // Forward the very last prompt token
    int current_pos = (int)input_tokens.size() - 1;
    if (current_pos >= max_context_window) return "";

    next_token = forward(input_tokens.back(), current_pos);

    // Calculate TTFT
    auto first_token_time = std::chrono::high_resolution_clock::now();
    double ttft = std::chrono::duration<double, std::milli>(first_token_time - start_time).count();
    std::cout << "\n[Metrics] TTFT: " << ttft << " ms" << std::endl;

    int generated_count = 0;
    auto decode_start = std::chrono::high_resolution_clock::now();

    // --- DECODE PHASE (Autoregressive Loop) ---
    for (int step = current_pos + 1; step < current_pos + max_tokens; ++step) {
        // Prevent Buffer Overflow in JIT Kernels
        if (step >= max_context_window) {
            std::cout << "\n[Warning] Context window limit (2048) reached. Stopping generation." << std::endl;
            break;
        }

        // Decode the integer back to text and stream to console smoothly
        std::string part = tokenizer.decode(next_token);
        result += part;
        std::cout << part << std::flush;

        generated_count++;

        // Stop conditions: <eos> (1), <pad> (0), or <end_of_turn> (107)
        if (next_token == 1 || next_token == 0 || next_token == 107) {
            break;
        }

        // Feed the predicted token back in to get the next one
        next_token = forward(next_token, step);
    }

    // Calculate Generation Speed
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_decode_time = std::chrono::duration<double>(end_time - decode_start).count();
    std::cout << "\n\n[Metrics] Speed: " << (generated_count / total_decode_time) << " tokens/sec" << std::endl;

    // Check Memory Footprint
    print_ram_usage();

    std::cout << std::endl;
    return result;
}
