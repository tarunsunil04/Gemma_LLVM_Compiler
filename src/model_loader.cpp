#include "include/model_loader.h"

#include <iostream>
#include <stdexcept>


/* This is responsible for loading and parsing the SAFETENSORS file that you generate using the python script. This is surprisingly straightforward. This is a testament to
how the SAFETENSORS are formatted really. Guess the idea of a JSON for weights was pretty good. */

ModelLoader::ModelLoader(const std::vector<std::string>& shard_paths) {
    for (const auto& path : shard_paths) {
        map_and_parse_shard(path);
    }
}

void ModelLoader::map_and_parse_shard(const std::string& path) {
    Shard shard;
    shard.hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (shard.hFile == INVALID_HANDLE_VALUE) throw std::runtime_error("File not found: " + path);

    //create Mapping Object
    shard.hMapping = CreateFileMapping(shard.hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!shard.hMapping) {
        CloseHandle(shard.hFile);
        throw std::runtime_error("Failed to create file mapping for: " + path);
    }

    //map the View into Process Address Space
    shard.base_ptr = MapViewOfFile(shard.hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!shard.base_ptr) {
        CloseHandle(shard.hMapping);
        CloseHandle(shard.hFile);
        throw std::runtime_error("Failed to map view of file: " + path);
    }

    parse_buffer(shard.base_ptr);
    shards.push_back(shard);
}

void ModelLoader::parse_buffer(void* ptr) {
    uint64_t header_size = *(uint64_t*)ptr; // Usual Safetensors format ---> First 8 bytes = N (header size) ---> N bytes of JSON
    std::string header_json((char*)ptr + 8, header_size);

    auto json_data = nlohmann::json::parse(header_json);

    for (auto& [name, info] : json_data.items()) {
        if (name == "__metadata__") continue;

        Tensor t;
        t.dtype = info["dtype"];
        t.shape = info["shape"].get<std::vector<int64_t>>();

        uint64_t start_offset = info["data_offsets"][0]; 
        t.data = (uint8_t*)ptr + 8 + header_size + start_offset; // Calculate absolute pointer: Base + 8 (size field) + Header Length + Tensor Offset

        tensor_map[name] = t;
    }
}

Tensor ModelLoader::get_tensor(const std::string& name) {
    auto it = tensor_map.find(name);
    if (it != tensor_map.end()) {
        return it->second;
    }
    return { nullptr, {}, "" };
}

ModelLoader::~ModelLoader() {
    for (auto& shard : shards) {
        if (shard.base_ptr) UnmapViewOfFile(shard.base_ptr);
        if (shard.hMapping) CloseHandle(shard.hMapping);
        if (shard.hFile) CloseHandle(shard.hFile);
    }
}