#include "include/gemma_tokenizer.h"

std::vector<int> GemmaTokenizer::encode_with_bos(const std::string& text) {
    std::vector<int> tokens;
    processor.Encode(text, &tokens);
    tokens.insert(tokens.begin(), 2);
    return tokens;
}