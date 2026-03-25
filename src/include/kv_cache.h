#pragma once
#include <vector>

class KVCache {
public:
    KVCache(const GemmaConfig& config, int max_seq_len)
        : config(config), max_seq_len(max_seq_len), current_pos(0) {

        size_t total_elements = config.num_layers * 2 * config.num_kv_heads * config.head_dim * max_seq_len;
        cache_data.resize(total_elements, 0.0f);
    }

    //returns the pointer to the start of the K or V cache for a specific layer or position, for better naviagtion of the cache.
    float* get_layer_kv(int layer, bool is_value, int pos) {
        size_t layer_offset = layer * 2 * config.num_kv_heads * config.head_dim * max_seq_len;
        size_t kv_offset = (is_value ? 1 : 0) * config.num_kv_heads * config.head_dim * max_seq_len;
        size_t pos_offset = pos * config.num_kv_heads * config.head_dim;

        return &cache_data[layer_offset + kv_offset + pos_offset];
    }

    void increment_pos() { current_pos++; }
    int get_current_pos() { return current_pos; }

private:
    const GemmaConfig& config;
    int max_seq_len;
    int current_pos;
    std::vector<float> cache_data;
};