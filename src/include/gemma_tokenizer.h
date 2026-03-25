#pragma once
#include <sentencepiece_processor.h>
#include <string>
#include <iostream>
#include <vector>

/* This is where the tokenizer for the Gemma model is intitialized. Building it from scratch is also cool, and could very much be done, but for now I've used the sentencepiece
library for this. (Thank god they have a CPP library!) */

class GemmaTokenizer {
public:
    GemmaTokenizer(const std::string& model_path) {
        auto status = processor.Load(model_path);
        if (!status.ok()) {
            throw std::runtime_error("Failed to load SentencePiece model: " + status.ToString());
        }
    }

    std::vector<int> encode(const std::string& text) {
        std::vector<int> tokens;
        processor.Encode(text, &tokens);
        return tokens;
    }

    std::string decode(int token_id) {
        std::string text;
        processor.Decode({ token_id }, &text);
        return text;
    }

    std::vector<int> encode_with_bos(const std::string& text);

private:
    sentencepiece::SentencePieceProcessor processor;
};