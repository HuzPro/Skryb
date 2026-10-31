// The native engines' weight loader: a file's tensors from `start` on, straight into VRAM.
#pragma once
#include "gpu.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// Reader threads fill a ring of ordinary (pageable) buffers and the calling thread copies each one to
// VRAM in order. Pinning host memory costs ~230 ms on first use under ROCm, while pageable copies
// still run at ~20 GB/s, well above what the disk or the page cache delivers, so reading is the only
// cost left. Unbuffered reads skip the page cache: steady disk speed, but they never warm the cache; they
// need a 4096-aligned start.
inline void stream_to_device(const char* path, uint64_t start, bool unbuffered, int readers, char* dev, uint64_t bytes) {
    constexpr uint64_t kStep = 16ull << 20;
    constexpr int kSlots = 16;
    const uint64_t chunks = (bytes + kStep - 1) / kStep;
    char* ring = (char*)VirtualAlloc(nullptr, kSlots * kStep, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!ring) throw std::runtime_error("cannot allocate the read ring");
    std::mutex mu;
    std::condition_variable cv;
    std::vector<int64_t> filled(kSlots, -1);  // chunk each slot holds, -1 when free
    uint64_t consumed = 0;
    std::string error;
    auto fail = [&](const std::string& why) {
        std::lock_guard<std::mutex> lk(mu);
        if (error.empty()) error = why;
        cv.notify_all();
    };
    auto reader = [&](int r) {
        HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                               unbuffered ? FILE_FLAG_NO_BUFFERING : FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return fail(std::string("cannot open ") + path);
        for (uint64_t c = r; c < chunks; c += readers) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return c < consumed + kSlots || !error.empty(); });
                if (!error.empty()) break;
            }
            const int slot = (int)(c % kSlots);
            const uint64_t at = start + c * kStep, want = std::min(kStep, bytes - c * kStep);
            OVERLAPPED ov{};
            ov.Offset = (DWORD)at;
            ov.OffsetHigh = (DWORD)(at >> 32);
            DWORD got = 0;
            const DWORD ask = (DWORD)(unbuffered ? (want + 4095) / 4096 * 4096 : want);
            if (!ReadFile(h, ring + slot * kStep, ask, &got, &ov) || got < want) {
                fail("short read");
                break;
            }
            std::lock_guard<std::mutex> lk(mu);
            filled[slot] = (int64_t)c;
            cv.notify_all();
        }
        CloseHandle(h);
    };
    std::vector<std::thread> pool;
    for (int r = 0; r < readers; r++) pool.emplace_back(reader, r);
    for (uint64_t c = 0; c < chunks; c++) {
        const int slot = (int)(c % kSlots);
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return filled[slot] == (int64_t)c || !error.empty(); });
            if (!error.empty()) break;
        }
        const hipError_t e = hipMemcpy(dev + c * kStep, ring + slot * kStep, std::min(kStep, bytes - c * kStep),
                                       hipMemcpyHostToDevice);
        if (e != hipSuccess) {
            fail(std::string("hipMemcpy: ") + hipGetErrorString(e));
            break;
        }
        std::lock_guard<std::mutex> lk(mu);
        filled[slot] = -1;
        consumed = c + 1;
        cv.notify_all();
    }
    for (auto& t : pool) t.join();
    VirtualFree(ring, 0, MEM_RELEASE);
    if (!error.empty()) throw std::runtime_error(error);
}
