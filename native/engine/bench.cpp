// qasr-bench <qasr.bin> <corpus.jsonl> <out.jsonl> [limit]   transcribe the benchmark set
// qasr-bench <qasr.bin> <file.wav>                             transcribe one file
#include "qasr.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// 16-bit PCM mono WAV, the format the app saves, scaled like the reference (/32768).
std::vector<float> read_wav(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), {});
    if (raw.size() < 12 || std::memcmp(raw.data(), "RIFF", 4)) throw std::runtime_error("not a wav: " + path);
    size_t p = 12;
    while (p + 8 <= raw.size()) {
        uint32_t len;
        std::memcpy(&len, raw.data() + p + 4, 4);
        if (!std::memcmp(raw.data() + p, "data", 4)) {
            const int16_t* s = (const int16_t*)(raw.data() + p + 8);
            size_t n = std::min<size_t>(len, raw.size() - p - 8) / 2;
            std::vector<float> out(n);
            for (size_t i = 0; i < n; i++) out[i] = s[i] / 32768.0f;
            return out;
        }
        p += 8 + len + (len & 1);
    }
    throw std::runtime_error("no data chunk: " + path);
}

// The string value of "key" in one JSON line (handles the escapes json.dumps writes for paths).
std::string json_str(const std::string& line, const std::string& key) {
    size_t at = line.find("\"" + key + "\": \"");
    if (at == std::string::npos) return "";
    std::string out;
    for (size_t i = at + key.size() + 5; i < line.size() && line[i] != '"'; i++) {
        if (line[i] != '\\') { out += line[i]; continue; }
        char c = line[++i];
        if (c == 'u') {
            unsigned cp = std::stoul(line.substr(i + 1, 4), nullptr, 16);
            i += 4;
            if (cp < 0x80) out += (char)cp;
            else if (cp < 0x800) { out += (char)(0xc0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3f)); }
            else { out += (char)(0xe0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3f)); out += (char)(0x80 | (cp & 0x3f)); }
        } else out += c == 'n' ? '\n' : c == 't' ? '\t' : c;
    }
    return out;
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
        else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += (char)c;
    }
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: qasr-bench <qasr.bin> <corpus.jsonl> <out.jsonl> [limit] | <qasr.bin> <file.wav>\n");
        return 2;
    }
    char err[512];
    const double t0 = now_ms();
    QasrEngine* e = qasr_load(argv[1], err, sizeof err);
    if (!e) { std::fprintf(stderr, "load failed: %s\n", err); return 1; }
    std::printf("load %.1f ms %s\n", now_ms() - t0, qasr_stats(e));
    std::vector<char> text(1 << 20);
    auto run = [&](const std::vector<float>& pcm, double& ms) {
        const double t = now_ms();
        int n = qasr_transcribe(e, pcm.data(), (int)pcm.size(), text.data(), (int)text.size());
        ms = now_ms() - t;
        if (n < 0) throw std::runtime_error(text.data());
        return std::string(text.data(), n);
    };
    try {
        double ms;
        if (argc == 3) {
            auto pcm = read_wav(argv[2]);
            run(pcm, ms);  // warm-up
            std::string t = run(pcm, ms);
            std::printf("%.1f ms %s\n%s\n", ms, qasr_stats(e), t.c_str());
            qasr_free(e);
            return 0;
        }
        std::ifstream in(argv[2]);
        std::ofstream out(argv[3], std::ios::binary);
        const int limit = argc > 4 ? std::atoi(argv[4]) : 0;
        std::vector<std::string> lines;
        for (std::string line; std::getline(in, line);) if (!line.empty()) lines.push_back(line);
        if (limit) lines.resize(std::min<size_t>(limit, lines.size()));
        run(read_wav(json_str(lines[0], "audio")), ms);  // warm-up, not timed
        int i = 0;
        for (const auto& line : lines) {
            auto pcm = read_wav(json_str(line, "audio"));
            std::string t = run(pcm, ms);
            out << "{\"id\": \"" << json_escape(json_str(line, "id")) << "\", \"seconds\": " << ms / 1000.0
                << ", \"audio_seconds\": " << pcm.size() / 16000.0 << ", \"stats\": " << qasr_stats(e)
                << ", \"text\": \"" << json_escape(t) << "\"}\n";
            out.flush();
            std::printf("%d/%zu %6.1fs audio %8.1f ms %s\n", ++i, lines.size(), pcm.size() / 16000.0, ms, qasr_stats(e));
            std::fflush(stdout);
        }
    } catch (const std::exception& x) {
        std::fprintf(stderr, "error: %s\n", x.what());
        qasr_free(e);
        return 1;
    }
    qasr_free(e);
    return 0;
}
