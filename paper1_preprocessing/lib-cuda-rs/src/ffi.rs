//extern crate candle;
//use std::ffi::c_void;
use std::ffi::{c_int, c_void};
//use core::ffi::{c_int, c_void};

extern "C" {

    pub(crate) fn ffi_hello_world_add();

    pub(crate) fn ffi_matmult_sans_batch(
        a_ptr:    *const c_void,
        b_ptr:    *const c_void,
        c_ptr:    *const c_void,
        a_height: c_int,
        a_width:  c_int,
        b_width:  c_int);

    pub(crate) fn ffi_softmax_vec(
        input_ptr:  *const c_void,
        output_ptr: *const c_void,
        n:          c_int);

    pub(crate) fn ffi_softmax(
        input_ptr: *const c_void,
        output_ptr: *const c_void,
        n:          c_int,   // Nb of element for a SoftMax
        b:          c_int);  // Batch size

    pub(crate) fn ffi_interpolate_2d(
        input_ptr: *const c_void,
        output_ptr: *const c_void,
        source_h: c_int,
        source_w: c_int,
        target_h: c_int,
        target_w: c_int,
        interp_type: c_int,
        b: c_int);


    pub(crate) fn ffi_attn_cuda(
        q_ptr: *const c_void,
        k_ptr: *const c_void,
        v_ptr: *const c_void,
        o_ptr: *const c_void,
        n:      c_int,
        d:      c_int,
        b:      c_int,
        attn_type: c_int); 

}
