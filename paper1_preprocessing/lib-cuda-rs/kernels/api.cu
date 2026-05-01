#include <iostream>
#include "kernels.h"
//#include "kernel_helpers.h"
//#include "flash_fwd_launch_template.h"

//#include <cuda_runtime.h>

//#include <cuda_runtime_api.h>


///////////////
// Interface //
///////////////

extern "C" void ffi_hello_world_add() {
    cudaStream_t stream = 0; // Use the default stream.
    run_hello_world_add(stream); }


//void run_matmult(void* a_ptr, void* b_ptr, void* c_ptr, int a_height, int a_width, int b_width);
extern "C" void ffi_matmult_sans_batch(
    void *a_ptr,
    void *b_ptr,
    void *c_ptr,
    int a_height,
    int a_width,
    int b_width) {
    run_matmult_sans_batch(a_ptr, b_ptr, c_ptr, a_height, a_width, b_width); }

extern "C" void ffi_softmax_vec(void *input_ptr, void *output_ptr, int n) {
    run_softmax_vec(input_ptr, output_ptr, n); }


extern "C" void ffi_softmax(
    void  *input_ptr,
    void  *output_ptr,
    int   n,
    int   b) {
        std::cout << "Appel à api.cu > ffi_softmax" << std::endl;
        run_softmax(
            input_ptr,
            output_ptr,
            n,
            b); }




extern "C" void ffi_interpolate_2d(
    void  *input_ptr,
    void  *output_ptr,
    int source_h,
    int source_w,
    int target_h,
    int target_w,
    int interp_type,
    int b) {
        //std::cout << "Appel à api.cu > ffi_interpolate_2d" << std::endl;
        run_interpolate_2d(
            input_ptr,
            output_ptr,
            source_h,
            source_w,
            target_h,
            target_w,
            interp_type,
            b); }

        

extern "C" void ffi_attn_cuda(
    void *q_ptr,
    void *k_ptr,
    void *v_ptr,
    void *o_ptr,
    int   n,
    int   d,
    int   b,
    int attn_type) {
    //std::cout << "Appel à api.cu > ffi_attn_cuda (attn_type = " << attn_type << ")" << std::endl;

    // Piping test
    if (attn_type == 0) {return;}

    run_attn_cuda(q_ptr, k_ptr, v_ptr, o_ptr, n, d, b, attn_type);
}