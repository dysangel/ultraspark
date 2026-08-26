// state_restore.cpp — issue #8 part 2 gate: restore a state blob, decode one
// more token, print/save next-token logits. Two modes:
//   ref     <gguf> <prompt> <extra_tok> <out.logits>   (decode prompt + extra tok)
//   restore <gguf> <blob.bin> <extra_tok> <out.logits> (set_data blob, decode extra tok)
// Build: c++ -O2 -std=c++17 -I ~/Projects/llama.cpp-ultraspark2/include \
//        state_restore.cpp ~/Projects/llama.cpp-ultraspark2/build-metal/bin/libllama.dylib -o state_restore
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static float *run_decode_extra(llama_context *ctx, std::vector<llama_token> &prompt,
                               const char *blob, llama_token extra) {
    if (blob) {
        std::ifstream f(blob, std::ios::binary | std::ios::ate);
        if (!f) { fprintf(stderr, "open blob failed\n"); exit(1); }
        size_t sz = f.tellg(); f.seekg(0);
        std::vector<uint8_t> buf(sz);
        f.read((char *)buf.data(), sz);
        // note: on a fresh (undecoded) context get_size_ext returns a minimal
        // size (604 here) — session-style restore via set_data_ext is the
        // intended path and grows the KV cells itself.
        if (llama_state_seq_set_data_ext(ctx, buf.data(), buf.size(), 0, 0) != sz) {
            fprintf(stderr, "set_data failed\n"); exit(1);
        }
        fprintf(stderr, "restored %zu bytes\n", sz);
    } else {
        llama_batch batch = llama_batch_get_one(prompt.data(), (int)prompt.size());
        if (llama_decode(ctx, batch)) { fprintf(stderr, "decode failed\n"); exit(1); }
    }
    llama_batch one = llama_batch_get_one(&extra, 1);
    if (llama_decode(ctx, one)) { fprintf(stderr, "extra decode failed\n"); exit(1); }
    return llama_get_logits_ith(ctx, 0);
}

int main(int argc, char **argv) {
    if (argc != 6) { fprintf(stderr, "usage: %s ref|restore <gguf> <prompt-or-blob> <extra_tok> <out.logits>\n", argv[0]); return 1; }
    std::string mode = argv[1];
    llama_model_params mp = llama_model_default_params();
    llama_model *model = llama_model_load_from_file(argv[2], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 8192;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    llama_token extra = (llama_token)atoi(argv[4]);

    std::vector<llama_token> prompt;
    const char *blob = nullptr;
    if (mode == "ref") {
        std::string p = argv[3];
        std::vector<llama_token> toks(p.size() + 8);
        int n = llama_tokenize(vocab, p.c_str(), p.size(), toks.data(), toks.size(), true, false);
        if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
        prompt.assign(toks.begin(), toks.begin() + n);
        fprintf(stderr, "tokens: %d\n", n);
    } else {
        blob = argv[3];
    }

    float *logits = run_decode_extra(ctx, prompt, blob, extra);
    int32_t nv = llama_vocab_n_tokens(vocab);
    std::ofstream g(argv[5]);
    g.write((char *)logits, sizeof(float) * nv);
    fprintf(stderr, "wrote %s (%d logits)\n", argv[5], nv);
    return 0;
}
