#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef GGML_USE_HIPBLAS
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_API GGML_CALL ggml_backend_t ggml_backend_cuda_init(int device, const void * params, const void * model);

GGML_API GGML_CALL bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_API GGML_CALL int  ggml_backend_cuda_get_device_count(void);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);
GGML_API GGML_CALL bool ggml_backend_cuda_is_integrated(int device);

GGML_API GGML_CALL bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
// Complete allocations deferred by GGML_Z6_NUMA_PINNED, after expert placement/load.
GGML_API GGML_CALL bool ggml_backend_cuda_host_register_pending(void);
GGML_API GGML_CALL void ggml_backend_cuda_host_defer(bool defer);
GGML_API GGML_CALL void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_API GGML_CALL void ggml_backend_cuda_log_set_callback(ggml_log_callback log_callback, void * user_data);

GGML_API GGML_CALL void ggml_backend_cuda_invalidate_graphs(const void * model);

// Z6: small explicit pinned staging, independent of GGML_CUDA_NO_PINNED.
GGML_API GGML_CALL void * ggml_backend_cuda_z6_host_alloc(size_t bytes);
// caldi dinamici: copie verso la VRAM su uno stream proprio, in parallelo al calcolo
GGML_API GGML_CALL void ggml_backend_cuda_z6_carica_async(ggml_backend_t backend, struct ggml_tensor * dst, size_t offset, const void * src, size_t size);
GGML_API GGML_CALL void ggml_backend_cuda_z6_carica_attendi(ggml_backend_t backend);
GGML_API GGML_CALL void ggml_backend_cuda_z6_host_free(void * ptr);
GGML_API GGML_CALL ggml_backend_event_t ggml_backend_cuda_z6_event_new(ggml_backend_t backend);
GGML_API GGML_CALL void ggml_backend_cuda_z6_event_record(ggml_backend_event_t event);
GGML_API GGML_CALL bool ggml_backend_cuda_z6_event_ready(ggml_backend_event_t event);
GGML_API GGML_CALL void ggml_backend_cuda_z6_get_async(ggml_backend_t backend, const struct ggml_tensor * src, void * dst);
GGML_API GGML_CALL void ggml_backend_cuda_z6_set_async(ggml_backend_t backend, struct ggml_tensor * dst, const void * src);

GGML_API GGML_CALL size_t ggml_cuda_quantize(int device, enum ggml_type type, const float * src, void * dst, int64_t nrows, int64_t n_per_row, int64_t nslice, const float * imatrix);
#ifdef  __cplusplus
}
#endif
