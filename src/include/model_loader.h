#define NOMINMAX

#pragma once
#include <windows.h>
#include <string>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>

/* Responsible for loading the model, as shards, which is the norm for downloading models as parquets. Also responsible for reading the model (or parsing if you're a stickler 
for that kind of detail). SAFETENSORS are basically giant JSON's. */

struct Tensor {
    void* data;
    std::vector<int64_t> shape;
    std::string dtype;
};

class ModelLoader {
public:
    ModelLoader(const std::vector<std::string>& shard_paths);
    ~ModelLoader();

    Tensor get_tensor(const std::string& name);

private:
    struct Shard {
        HANDLE hFile;
        HANDLE hMapping;
        void* base_ptr;
    };

    std::vector<Shard> shards;
    std::map<std::string, Tensor> tensor_map;

    void parse_buffer(void* ptr);
    void map_and_parse_shard(const std::string& path);
};