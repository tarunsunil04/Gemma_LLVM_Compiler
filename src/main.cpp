#include "include/jit.h"
#include "include/kernel_generator.h"
#include "include/model_loader.h"
#include "include/kv_cache.h"
#include "include/gemma_engine.h"

#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <memory>

#include <windows.h>
#include <psapi.h>

const int BOS_ID = 2;
const int START_TURN_ID = 106;
const int END_TURN_ID = 107;
const int USER_ID = 1645;
const int MODEL_ID = 2516;
const int NEWLINE_ID = 108;

struct ChatHistory {
    std::vector<int> tokens;
    GemmaTokenizer* tokenizer;
    int max_context = 1500; //<-- Leaves 500 tokens for the model's response (2048 total) This makes the response a lot smaller, sure, but this is to just show how it works.

    ChatHistory(GemmaTokenizer* tok) : tokenizer(tok) {
        tokens.push_back(BOS_ID);
    }

    void add_user_message(const std::string& text) {
        tokens.push_back(START_TURN_ID);
        tokens.push_back(USER_ID);
        tokens.push_back(NEWLINE_ID);

        std::vector<int> text_tokens = tokenizer->encode(text);
        tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());

        tokens.push_back(END_TURN_ID);
        tokens.push_back(NEWLINE_ID);

        // Prep the model to answer
        tokens.push_back(START_TURN_ID);
        tokens.push_back(MODEL_ID);
        tokens.push_back(NEWLINE_ID);

        enforce_context_limit();
    }

    void add_model_response(const std::string& text) {
        std::vector<int> text_tokens = tokenizer->encode(text);
        tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());

        tokens.push_back(END_TURN_ID);
        tokens.push_back(NEWLINE_ID);
    }

    void enforce_context_limit() {
        // If the history gets too long, we aggressively chop off the oldest 
        // user-model interaction to prevent a JIT buffer overflow.
        if (tokens.size() > max_context) {
            std::cout << "\n[System] Context limit reached. Forgetting oldest messages..." << std::endl;

            // Keep the BOS token, erase a chunk of history, and hope we land cleanly
            // (A production app would search for the exact END_TURN_ID to slice cleanly)
            int drop_amount = tokens.size() - max_context + 200;
            tokens.erase(tokens.begin() + 1, tokens.begin() + drop_amount);
        }
    }
};


/* For now, I've used std::filesystem for handling paths relatively. It's always better to use the absolute paths when passing a path on to a loader class. If you're
sure about your paths and stuff, you can get rid of this and give in your own absolute path. Or better yet, you could revamp the path logic. */

/* This is also recommended, if you're using CMake to build and setup the project. (This was done in VS 2022, so having VS with C++ desktop development enabled was quite helpful.) */

/* The format for the Gemma-2B-it (instruction-tuned) is as below. NOTE THAT THIS DOES NOT WORK IF YOU USE THE Gemma 2B model as is, as it isn't instruction tuned. You can't ask it questions. */

#include <filesystem>

int main() {
    std::vector<std::string> shards = {
        std::filesystem::absolute("../../../src/safetensors/model_dir/model-00001-of-00002.safetensors").string(),
        std::filesystem::absolute("../../../src/safetensors/model_dir/model-00002-of-00002.safetensors").string(),
    };

    std::cout << "The current working directory is: " << std::filesystem::current_path() << std::endl;
    /*std::cout << "" << std::endl;*/

    try {
        auto engine = std::make_unique<GemmaEngine>(shards, std::filesystem::absolute("../../../src/safetensors/model_dir/tokenizer.model").string());

        ChatHistory chat(&engine->tokenizer);

        while (true) {
            std::string user_input;
            std::cout << "\n[You]: ";
            std::getline(std::cin, user_input);

            if (user_input == "exit" || user_input == "quit") break;
            if (user_input.empty()) continue;
            chat.add_user_message(user_input);
            std::cout << "[Gemma]: " << std::flush;
            std::string response = engine->generate_from_tokens(chat.tokens, 500);
            chat.add_model_response(response);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Whoops: " << e.what() << std::endl;
    }
    return 0;
}