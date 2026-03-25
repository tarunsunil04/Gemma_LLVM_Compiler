#pragma once
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

//These details are pretty much straight from the Gemma paper.
//Here it is, if you want to have a look: https://arxiv.org/abs/2403.08295

struct GemmaConfig {
    int hidden_size         = 2048;
    int intermediate_size   = 16384;
    int num_layers          = 18;
    int num_heads           = 8;
    int num_kv_heads        = 1;
    int head_dim            = 256;
    float rms_norm_eps      = 1e-6f;
    int vocab_size          = 256128;

    static GemmaConfig from_file(const std::string& path) {
        std::ifstream f(path);
        
        auto j = nlohmann::json::parse(f);
        
        GemmaConfig c;

        c.hidden_size           = j["hidden_size"];
        c.intermediate_size     = j["intermediate_size"];
        c.num_layers            = j["num_hidden_layers"];
        c.num_heads             = j["num_attention_heads"];
        c.num_kv_heads          = j.value("num_key_value_heads", 1);
        c.rms_norm_eps          = j.value("rms_norm_eps", 1e-6f);
        c.vocab_size            = j["vocab_size"];

        return c;
    }
};