#ifndef _KERNELS_H_
#define _KERNELS_H_

/*
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include<stdlib.h>
#include<stdint.h>
*/

//#include <iostream>

void run_hello_world_add(cudaStream_t stream);

void run_matmult_sans_batch(void* a_ptr, void* b_ptr, void* c_ptr, int a_height, int a_width, int b_width);

void run_softmax_vec(void *input_ptr, void *output_ptr, int n);

void run_softmax(void *input_ptr, void *output_ptr, int n, int b);

void run_interpolate_2d(
    void *input_ptr,
    void *output_ptr,
    int source_h,
    int source_w,
    int target_h,
    int target_w,
    int interp_type,
    int b);

void run_attn_cuda(
    void *q_ptr,
    void *k_ptr,
    void *v_ptr,
    void *o_ptr,
    int   n,
    int   d,
    int   b,
    int attn_type);


#endif
