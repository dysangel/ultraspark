// state_dump.cpp — decode a prompt, dump llama.cpp's own KV state blob (issue #8).
// Build: c++ -O2 -std=c++17 -I ~/Projects/llama.cpp-ultraspark2/include \
//        state_dump.cpp ~/Projects/llama.cpp-ultraspark2/build-metal/bin/libllama.dylib -o state_dump
// Usage: state_dump <gguf> <prompt> <out.bin>
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s <gguf> <prompt> <out.bin>\n", argv[0]); return 1; }
    llama_model_params mp = llama_model_default_params();
    llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 8192;
    cp.n_batch = cp.n_ubatch = cp.n_ctx;  // whole-prompt decode: the blob must
    // cover the ENTIRE prompt (chunked single-shot decode asserts; e2e rule:
    // never emit a partial-coverage blob — issue #10)
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    std::string prompt = argv[2];
    const llama_vocab *vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(prompt.size() + 8);
    int n = llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), toks.size(), true, false);
    if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    toks.resize(n);
    fprintf(stderr, "tokens: %d\n", n);

    llama_batch batch = llama_batch_get_one(toks.data(), n);
    if (llama_decode(ctx, batch)) { fprintf(stderr, "decode failed\n"); return 1; }

    const size_t sz = llama_state_seq_get_size_ext(ctx, 0, 0);
    fprintf(stderr, "state size: %zu bytes for %d tokens\n", sz, n);
    std::vector<uint8_t> buf(sz);
    if (llama_state_seq_get_data_ext(ctx, buf.data(), buf.size(), 0, 0) != sz) { fprintf(stderr, "get failed\n"); return 1; }
    std::ofstream f(argv[3], std::ios::binary);
    f.write((char *)buf.data(), buf.size());
    // also dump next-token logits for the tolerance test (#10)
    float *logits = llama_get_logits_ith(ctx, n - 1);
    int32_t nv = llama_vocab_n_tokens(vocab);
    std::vector<std::pair<float, int>> top;
    for (int32_t i = 0; i < nv; i++) top.push_back({logits[i], i});
    std::string lit = argv[3] + std::string(".logits");
    std::ofstream g(lit);
    g.write((char *)logits, sizeof(float) * nv);
    fprintf(stderr, "wrote %s (%d logits)\n", lit.c_str(), nv);
    return 0;
}
