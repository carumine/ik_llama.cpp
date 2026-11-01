#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

static void check(bool ok, const char * why) {
    if (!ok) { fprintf(stderr, "FAIL %s\n", why); std::exit(1); }
}

// Read the same source as the full-model verification, without privileged move_pages.
static void placement(void * ptr, int node, size_t pages) {
    std::ifstream maps("/proc/self/numa_maps");
    std::string line;
    bool found = false;
    while (std::getline(maps, line)) {
        std::istringstream fields(line);
        uintptr_t addr;
        fields >> std::hex >> addr;
        if (addr != (uintptr_t) ptr) continue;
        const auto expected = "N" + std::to_string(node) + "=" + std::to_string(pages);
        check(line.find("bind:" + std::to_string(node)) != std::string::npos, "bind policy");
        check(line.find(expected) != std::string::npos, "resident pages on assigned node");
        printf("%s\n", line.c_str());
        found = true;
    }
    check(found, "mapping boundary found");
}

int main() {
    setenv("GGML_Z6_NUMA_EXPERTS", "1", 1);
    setenv("GGML_Z6_NUMA_PINNED", "1", 1);
    setenv("GGML_Z6_NUMA_SPLIT", "0.5", 1);
    unsetenv("GGML_CUDA_NO_PINNED");
    auto ctx = ggml_init({1024*1024, nullptr, true});
    auto t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 64, 512); // 8 MiB, page-aligned expert slices
    ggml_set_name(t, "blk.0.ffn_down_exps.weight");
    // Also exercise cancellation: an unregistered pending allocation must free cleanly.
    ggml_backend_cuda_host_defer(true);
    auto cancelled = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), 1 << 20);
    check(cancelled != nullptr, "cancelled allocation");
    ggml_backend_buffer_free(cancelled);
    auto buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cuda_host_buffer_type());
    check(buf != nullptr, "host weight allocation");
    std::memset(t->data, 1, ggml_nbytes(t)); // mirror defer-ple: populated before migration
    check(ggml_z6_numa_place_tensor(t), "mbind MOVE before register");
    placement(t->data, 0, 1024);
    placement((char *)t->data + (4 << 20), 1, 1024);
    cpu_set_t before, after;
    int mode_before = 0, mode_after = 0;
    unsigned long mask_before = 0, mask_after = 0;
    check(sched_getaffinity(0, sizeof(before), &before) == 0, "read initial affinity");
    check(syscall(SYS_get_mempolicy, &mode_before, &mask_before, 64, nullptr, 0) == 0, "read initial memory policy");
    check(ggml_backend_cuda_host_register_pending(), "cudaHostRegister after placement");
    check(sched_getaffinity(0, sizeof(after), &after) == 0 && CPU_EQUAL(&before, &after), "caller affinity restored");
    check(syscall(SYS_get_mempolicy, &mode_after, &mask_after, 64, nullptr, 0) == 0 &&
          mode_before == mode_after && mask_before == mask_after, "caller memory policy restored");
    ggml_backend_cuda_host_defer(false);
    placement(t->data, 0, 1024);
    placement((char *)t->data + (4 << 20), 1, 1024);
    auto gpu = ggml_backend_reg_init_backend_from_str("CUDA0");
    check(gpu != nullptr, "CUDA backend");
    auto device_ctx = ggml_init({1024*1024, nullptr, true});
    auto device_tensor = ggml_new_tensor_3d(device_ctx, GGML_TYPE_F32, 64, 64, 512);
    auto device_buffer = ggml_backend_alloc_ctx_tensors(device_ctx, gpu);
    check(device_buffer != nullptr, "device buffer");
    // A prompt upload must work across two adjacent CUDA registrations.
    ggml_backend_tensor_set(device_tensor, t->data, 0, ggml_nbytes(t));
    std::vector<unsigned char> output(ggml_nbytes(t));
    ggml_backend_tensor_get(device_tensor, output.data(), 0, output.size());
    check(std::memcmp(output.data(), t->data, output.size()) == 0, "whole tensor upload across registrations");
    ggml_backend_buffer_free(device_buffer);
    ggml_free(device_ctx);
    ggml_backend_free(gpu);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    printf("OK: cancelled load freed, mbind then pin, node residency and caller policies preserved, full upload correct\n");
}
