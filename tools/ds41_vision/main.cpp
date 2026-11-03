// tools/ds41_vision/main.cpp - ds41-vision: DeepSeek-V4.1's image encoder for the server's on-demand vision path.
//
// It speaks strata-vision's protocol (serve/server.py's Vision), so the server starts it, lends it GPU memory with the
// engine's VLEND and ends it the same way:
//   ds41-vision --mmproj <vision.bin> [--model <vision.json>] [--gpu] [--max-tokens N] [--no-warmup] [--measure]
//   -> READY, then per line:
//      ENC <in> <out>   in: DSR1 (int32 'DSR1', int32 width, int32 height, height x width x 3 RGB bytes; both sides
//                       multiples of the patch size: serve/ds41_vision.py resizes the picture first)
//                       out: one strata-vision record (int32 'SVE1', n, n_w, n_h, n_embd, then n x n_embd floats): the
//                       picture's whole span, [image_start] + ([row] * n_w + [image_newline]) * n_h + [image_end]
//                       -> OK <n> | ERR <reason>
//      QUIT
// --measure encodes the largest picture --max-tokens allows and prints MEM <bytes the card uses> 0 <card bytes>.
// The weights are tools/ds4/vision_pack.py's (Project Maya): vision.bin, its tensors and config in vision.json.
#include "ds4_vision.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct TensorRef {
    std::string dtype;
    uint64_t offset = 0, bytes = 0;
    std::vector<int64_t> shape;
};

struct Weights {
    std::ifstream bin;
    std::map<std::string, TensorRef> tensors;
    std::map<std::string, double> config;

    bool open(const std::string& bin_path, const std::string& json_path, std::string& err) {
        std::ifstream jf(json_path, std::ios::binary);
        if (!jf) {
            err = "cannot open " + json_path;
            return false;
        }
        std::stringstream ss;
        ss << jf.rdbuf();
        const std::string js = ss.str();
        // "config": {"key": number, ...} (flat; vision_pack.py writes numbers only)
        const size_t c0 = js.find("\"config\"");
        if (c0 != std::string::npos) {
            const size_t a = js.find('{', c0), b = js.find('}', a);
            if (a != std::string::npos && b != std::string::npos) {
                const std::string body = js.substr(a + 1, b - a - 1);
                static const std::regex kv("\"([^\"]+)\"\\s*:\\s*(-?[0-9.eE+]+)");
                for (auto it = std::sregex_iterator(body.begin(), body.end(), kv); it != std::sregex_iterator(); ++it)
                    config[(*it)[1].str()] = std::atof((*it)[2].str().c_str());
            }
        }
        // "tensors": [["name", "DTYPE", offset, bytes, [d0, d1, ...]], ...]
        static const std::regex te("\\[\\s*\"([^\"]+)\"\\s*,\\s*\"([A-Z0-9]+)\"\\s*,\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*,\\s*\\[([0-9,\\s]*)\\]\\s*\\]");
        for (auto it = std::sregex_iterator(js.begin(), js.end(), te); it != std::sregex_iterator(); ++it) {
            TensorRef t;
            t.dtype = (*it)[2].str();
            t.offset = std::strtoull((*it)[3].str().c_str(), nullptr, 10);
            t.bytes = std::strtoull((*it)[4].str().c_str(), nullptr, 10);
            std::stringstream dims((*it)[5].str());
            std::string d;
            while (std::getline(dims, d, ','))
                if (d.find_first_not_of(" \t\r\n") != std::string::npos) t.shape.push_back(std::atoll(d.c_str()));
            tensors[(*it)[1].str()] = t;
        }
        if (tensors.empty()) {
            err = json_path + ": no tensors";
            return false;
        }
        bin.open(bin_path, std::ios::binary);
        if (!bin) {
            err = "cannot open " + bin_path;
            return false;
        }
        return true;
    }
    double num(const std::string& k, double d) const {
        const auto it = config.find(k);
        return it == config.end() ? d : it->second;
    }
    bool read(const std::string& name, std::vector<int64_t>& shape, void* dst) {
        const auto it = tensors.find(name);
        if (it == tensors.end() || it->second.dtype != "BF16") return false;
        shape = it->second.shape;
        if (dst == nullptr) return true;
        bin.clear();
        bin.seekg((std::streamoff) it->second.offset);
        bin.read((char*) dst, (std::streamsize) it->second.bytes);
        return (bool) bin;
    }
};

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

struct Encoder {
    Weights w;
    strata::ds4::VisionGeo g;
    strata::ds4::VisionTower tower;
    std::vector<uint16_t> mark[3];   // image_start, image_newline, image_end: BF16 [out]
    uint8_t* room = nullptr;

    bool load(const std::string& bin, const std::string& json, int max_tokens, std::string& err) {
        if (!w.open(bin, json, err)) return false;
        g.layers = (int) w.num("vision.num_hidden_layers", 0);
        g.dim = (int) w.num("vision.hidden_size", 1024);
        g.heads = (int) w.num("vision.num_attention_heads", 16);
        g.inter = (int) w.num("vision.intermediate_size", 2816);
        g.patch = (int) w.num("vision.patch_size", 14);
        g.merge = (int) w.num("vision.downsample_ratio", 3);
        g.theta = (float) w.num("vision.rope_theta", 1e4);
        const int model_max = (int) w.num("vision.max_image_tokens", 1024);
        g.max_tokens = max_tokens > 0 ? std::min(max_tokens, model_max) : model_max;
        const char* names[3] = {"image_start", "image_newline", "image_end"};
        for (int i = 0; i < 3; ++i) {
            const auto it = w.tensors.find(names[i]);
            if (it == w.tensors.end() || it->second.dtype != "BF16" || it->second.shape.size() != 1) {
                err = std::string(names[i]) + ": not a BF16 row";
                return false;
            }
            g.out = (int) it->second.shape[0];
            mark[i].resize((size_t) g.out);
            std::vector<int64_t> shape;
            if (!w.read(names[i], shape, mark[i].data())) {
                err = std::string(names[i]) + ": read failed";
                return false;
            }
        }
        if (cudaSetDevice(0) != cudaSuccess) {
            err = "no CUDA device";
            return false;
        }
        auto reader = [this](const std::string& n, std::vector<int64_t>& shape, void* dst) { return w.read(n, shape, dst); };
        if (!tower.load(g, 0, reader, err)) return false;
        const cudaError_t e = cudaMalloc((void**) &room, tower.room_bytes());
        if (e != cudaSuccess) {
            err = std::string("the picture's room (") + std::to_string(tower.room_bytes() >> 20) + " MiB): " +
                  cudaGetErrorString(e);
            return false;
        }
        return true;
    }

    // rgb (height x width x 3) -> the span's rows as floats, n_w and n_h
    bool encode(const uint8_t* rgb, int width, int height, std::vector<float>& span, int& n_h, int& n_w,
                std::string& err) {
        std::vector<uint16_t> rows;
        if (!tower.encode(rgb, width, height, room, rows, n_h, n_w, err)) return false;
        const size_t D = (size_t) g.out;
        span.resize(((size_t) n_h * (n_w + 1) + 2) * D);
        float* o = span.data();
        auto put = [&](const uint16_t* r) {
            for (size_t k = 0; k < D; ++k) o[k] = bf16_to_f32(r[k]);
            o += D;
        };
        put(mark[0].data());
        for (int i = 0; i < n_h; ++i) {
            for (int j = 0; j < n_w; ++j) put(rows.data() + ((size_t) i * n_w + j) * D);
            put(mark[1].data());
        }
        put(mark[2].data());
        return true;
    }

    // the largest square picture whose span fits max_tokens (sides in patch-size multiples)
    void largest(int& width, int& height) const {
        const int cell = g.patch * g.merge;
        int n = 1;
        while ((n + 1) * (n + 2) + 2 <= g.max_tokens) ++n;
        width = height = n * cell;
    }
};

bool read_dsr(const std::string& path, std::vector<uint8_t>& rgb, int& width, int& height, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot open " + path;
        return false;
    }
    int32_t hdr[3];
    bool ok = std::fread(hdr, sizeof hdr, 1, f) == 1 && hdr[0] == 0x31525344 && hdr[1] > 0 && hdr[2] > 0 &&
              (int64_t) hdr[1] * hdr[2] <= (int64_t) 1 << 26;
    if (ok) {
        width = hdr[1];
        height = hdr[2];
        rgb.resize((size_t) width * height * 3);
        ok = std::fread(rgb.data(), 1, rgb.size(), f) == rgb.size();
    }
    std::fclose(f);
    if (!ok) err = "not a DSR1 picture: " + path;
    return ok;
}

bool write_sve(const std::string& path, const std::vector<float>& span, int n_w, int n_h, int n_embd, int& n,
               std::string& err) {
    n = (int) (span.size() / (size_t) n_embd);
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        err = "cannot write " + path;
        return false;
    }
    const int32_t hdr[5] = {0x31455653, n, n_w, n_h, n_embd};
    const bool ok = std::fwrite(hdr, sizeof hdr, 1, f) == 1 &&
                    std::fwrite(span.data(), sizeof(float), span.size(), f) == span.size();
    if (std::fclose(f) != 0 || !ok) {
        err = "short write: " + path;
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string bin, json;
    int max_tokens = 0;
    bool warmup = true, measure = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--mmproj" && i + 1 < argc) bin = argv[++i];
        else if (a == "--model" && i + 1 < argc) json = argv[++i];
        else if (a == "--max-tokens" && i + 1 < argc) max_tokens = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) ++i;   // (strata-vision's CPU option: the tower runs on the GPU)
        else if (a == "--no-warmup") warmup = false;
        else if (a == "--measure") measure = true;
        else if (a == "--gpu" || a == "--no-flash-attn") continue;
        else {
            std::fprintf(stderr, "ds41-vision: unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (bin.empty()) {
        std::fprintf(stderr, "usage: ds41-vision --mmproj vision.bin [--model vision.json] [--max-tokens N] "
                             "[--no-warmup] [--measure]\n");
        return 2;
    }
    if (json.empty() || json.size() < 5 || json.compare(json.size() - 5, 5, ".json") != 0) {
        const size_t dot = bin.rfind('.');
        json = (dot == std::string::npos ? bin : bin.substr(0, dot)) + ".json";
    }
    Encoder enc;
    std::string err;
    if (!enc.load(bin, json, max_tokens, err)) {
        std::printf("ERR %s\n", err.c_str());
        std::fflush(stdout);
        std::fprintf(stderr, "ds41-vision: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "ds41-vision: %d blocks, %d image tokens at most, %.0f MiB of weights in pinned RAM, "
                         "%.0f MiB of room on the GPU\n",
                 enc.g.layers, enc.g.max_tokens, (double) enc.tower.host_bytes() / (1 << 20),
                 (double) enc.tower.room_bytes() / (1 << 20));
    if (warmup || measure) {
        int w = 0, h = 0;
        if (measure) enc.largest(w, h);
        else w = h = enc.g.patch * enc.g.merge;
        std::vector<uint8_t> gray((size_t) w * h * 3, 127);
        std::vector<float> span;
        int n_h = 0, n_w = 0;
        if (!enc.encode(gray.data(), w, h, span, n_h, n_w, err)) {
            std::fprintf(stderr, "ds41-vision: warm-up: %s\n", err.c_str());
            return 1;
        }
        if (measure) {
            size_t free_b = 0, total_b = 0;
            if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 1;
            std::printf("MEM %zu 0 %zu\n", total_b - free_b, total_b);
            std::fflush(stdout);
            return 0;
        }
    }
    std::printf("READY\n");
    std::fflush(stdout);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream ss(line);
        std::string cmd, in, out;
        ss >> cmd;
        if (cmd == "QUIT") break;
        if (cmd != "ENC" || !(ss >> in >> out)) {
            std::printf("ERR expected ENC <in> <out> or QUIT\n");
            std::fflush(stdout);
            continue;
        }
        std::vector<uint8_t> rgb;
        std::vector<float> span;
        int w = 0, h = 0, n_h = 0, n_w = 0, n = 0;
        if (read_dsr(in, rgb, w, h, err) && enc.encode(rgb.data(), w, h, span, n_h, n_w, err) &&
            write_sve(out, span, n_w, n_h, enc.g.out, n, err))
            std::printf("OK %d\n", n);
        else
            std::printf("ERR %s\n", err.c_str());
        std::fflush(stdout);
    }
    return 0;
}
