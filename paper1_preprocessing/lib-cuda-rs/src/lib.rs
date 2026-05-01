mod ffi;

use std::convert::TryInto;

extern crate candle;
//extern crate candle_nn;
extern crate half;

//use candle::op::BackpropOp;
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::DevicePtr;
//use candle::cuda_backend::WrapErr;
//use candle::{CpuStorage, DType, Layout, Result, Shape, Tensor};
use candle::{CpuStorage, Layout, Shape}; //, DType};
use candle::{Result, Tensor};
use half::{bf16, f16};


/// Exécute un simple "hello world" additionnel sur GPU pour vérifier la connexion avec le backend CUDA.
pub fn cuda_hello_world_add() -> Result<()> {
    unsafe {
        ffi::ffi_hello_world_add();
    }
    return Ok(()); }

/// Applique un softmax sur un vecteur de f32 côté GPU.
/// **Note** : Cette version est une implémentation basique pour test, préférer `cuda_softmax` pour les tenseurs.
pub fn cuda_softmax_vec(input: &Vec<f32>) -> Result<Vec<f32>> {
    let n = input.len();

    let output: Vec<f32> = std::iter::repeat(0.0).take(n).collect();

    let input_slice = *&input;
    //let output_slice = output.clone;

    let input_ptr: *const core::ffi::c_void = input_slice.as_ptr() as *const core::ffi::c_void;
    let output_ptr: *const core::ffi::c_void = output.as_ptr() as *const core::ffi::c_void;

    unsafe {
        ffi::ffi_softmax_vec(input_ptr, output_ptr, n.try_into().unwrap());
    }
    return Ok(output); }


pub struct SSoftMax  {}

impl SSoftMax {
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        input: &candle::CudaStorage,
        input_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {

        let dev = input.device();
        let out_shape = input_l.shape().clone();

        let input = input.as_cuda_slice::<T>()?;
        let input = input.slice(input_l.start_offset()..);

        let elem_count = out_shape.elem_count();
        //println!("Debug:{}", elem_count);
        //let dst = unsafe { dev.alloc::<T>(elem_count) }.w()?;
        let dst = unsafe { dev.alloc::<T>(elem_count)? };
        let dims: Vec<i32> = input_l.dims().into_iter().map(|&x| x as i32).collect();
        let n : i32 = dims[dims.len() - 1];
        let b : i32 = dims.iter().rev().skip(1).rev().fold(1, |acc, x| acc * x);


        let stream = dev.cuda_stream(); 
        unsafe {
            let (input_ptr, _guard) = input.device_ptr(&stream);
            let (output_ptr, _guard) = dst.device_ptr(&stream);
            //let input_ptr = *input.device_ptr() as *const core::ffi::c_void;
            //let output_ptr = *dst.device_ptr() as *const core::ffi::c_void;
            ffi::ffi_softmax(
                input_ptr as *const core::ffi::c_void,
                output_ptr as *const core::ffi::c_void,
                n,
                b);

        }

        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, out_shape))
    }
}

impl candle::CustomOp1 for SSoftMax {
    fn name(&self) -> &'static str {
        "struct softmax"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for softmax")
    }

    fn cuda_fwd(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        match q.dtype() {
            candle::DType::F16 => self.cuda_fwd_t::<f16>(q, q_l),
            candle::DType::BF16 => self.cuda_fwd_t::<bf16>(q, q_l),
            candle::DType::F32 => self.cuda_fwd_t::<f32>(q, q_l),
            dt => candle::bail!("softmax is only supported for f16/bf16/f32 ({dt:?})"),
        }
    }
}

/// Applique un softmax sur la dernière dimension d'un tenseur CUDA.
pub fn cuda_softmax(t: &Tensor) -> Result<Tensor> {
    let op = SSoftMax {};
    let output = t.apply_op1(op)?;
    Ok(output)
}


// Produit matriciel sur les 2 dernières dimensions
pub struct SMatProdSansBatch  { }

impl SMatProdSansBatch {
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        a: &candle::CudaStorage,
        a_l: &Layout,
        b: &candle::CudaStorage,
        b_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {

        let dev = a.device();

        let a = a.as_cuda_slice::<T>()?;
        let b = b.as_cuda_slice::<T>()?;
        let a = a.slice(a_l.start_offset()..);
        let b = b.slice(b_l.start_offset()..);

        let dims_a: Vec<i32> = a_l.dims().into_iter().map(|&x| x as i32).collect();
        let dims_b: Vec<i32> = b_l.dims().into_iter().map(|&x| x as i32).collect();
        println!("debug dims a:{:?}", dims_a);
        println!("debug dims b:{:?}", dims_b);
        if dims_a.len() != 2 {println!("Erreur: A doit être de dimensions 2.");panic!("");}
        if dims_b.len() != 2 {println!("Erreur: B doit être de dimensions 2.");panic!("");}
        if dims_a[1] != dims_b[0] {println!("Erreur: A.n_cols = B.n_rows n'est pas vérifié.");panic!("");}
        let a_height = dims_a[0];
        let a_width = dims_a[1];
        let b_width = dims_b[1];

        let out_shape = Shape::from_dims(&[a_height as usize, b_width as usize]);
        let elem_count = out_shape.elem_count();
        //println!("Debug:{}", elem_count);
        let c = unsafe { dev.alloc::<T>(elem_count) }?;

        let stream = dev.cuda_stream();
        unsafe {
            let (a_ptr, _guard) = a.device_ptr(&stream);
            let (b_ptr, _guard) = b.device_ptr(&stream);
            let (c_ptr, _guard) = c.device_ptr(&stream);

            ffi::ffi_matmult_sans_batch(
                a_ptr as *const core::ffi::c_void,
                b_ptr as *const core::ffi::c_void,
                c_ptr as *const core::ffi::c_void,
                a_height,
                a_width,
                b_width) }

        let c = candle::CudaStorage::wrap_cuda_slice(c, dev.clone());
        Ok((c, out_shape))
    }
}

impl candle::CustomOp2 for SMatProdSansBatch {
    fn name(&self) -> &'static str {
        "struct matprod"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for matprod")
    }

    fn cuda_fwd(
        &self,
        a: &candle::CudaStorage,
        a_l: &Layout,
        b: &candle::CudaStorage,
        b_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        match a.dtype() {
            candle::DType::F16 => self.cuda_fwd_t::<f16>(a, a_l, b, b_l),
            candle::DType::BF16 => self.cuda_fwd_t::<bf16>(a, a_l, b, b_l),
            candle::DType::F32 => self.cuda_fwd_t::<f32>(a, a_l, b, b_l),
            dt => candle::bail!("matprod is only supported for f16/bf16/f32 ({dt:?})"),
        }
    }
}

/// Effectue un produit matriciel sur les 2 dernières dimensions de deux tenseurs CUDA.
/// **Précondition** : Les tenseurs doivent être 2D (HxW) et compatibles pour la multiplication (A.cols == B.rows).
pub fn cuda_matprod_sans_batch(
    a: &Tensor,
    b: &Tensor,
) -> Result<Tensor> {

    let op = SMatProdSansBatch {};
    let output = a.apply_op2(b,  op)?;
    Ok(output)
}



pub struct SInterp2D  {
    target_h: usize,
    target_w: usize,
    interp_type: usize,    // 0 = nearest, 1 = bilinear
}

impl SInterp2D {
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        input: &candle::CudaStorage,
        input_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {

        let dev = input.device();

        let input = input.as_cuda_slice::<T>()?;
        let input = input.slice(input_l.start_offset()..);

        let mut dims: Vec<usize> = Vec::from(input_l.dims());
        let n_dims = dims.len();
        if n_dims < 2 {
            panic!("Error in SInterp2D cuda_fwd_t(): Tensor must have at least 2 dimensions");
        }
        let source_h = dims[n_dims - 2];
        let source_w = dims[n_dims - 1];
        dims[n_dims - 2] = self.target_h;
        dims[n_dims - 1] = self.target_w;

        let out_shape = Shape::from_dims(&dims);
        let elem_count = out_shape.elem_count();
        //println!("Debug:{}", elem_count);
        let dst = unsafe { dev.alloc::<T>(elem_count)? };

        // Le nombre de blocs (le produit des dimensions au dessus de (H, W), y compris C les channels)
        let b : i32 = dims.into_iter().map(|x| x as i32).rev().skip(2).rev().fold(1, |acc, x| acc * x);
        let stream = dev.cuda_stream();

        unsafe {
            let (input_ptr, _guard) = input.device_ptr(&stream);
            let (output_ptr, _guard) = dst.device_ptr(&stream);
            ffi::ffi_interpolate_2d(
                input_ptr as *const core::ffi::c_void,
                output_ptr as *const core::ffi::c_void,
                source_h         as i32,
                source_w         as i32,
                self.target_h    as i32,
                self.target_w    as i32,
                self.interp_type as i32,
                b);
        }

        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, out_shape))
    }
}

impl candle::CustomOp1 for SInterp2D {
    fn name(&self) -> &'static str {
        "struct interpolation 2d"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for interpolation 2d")
    }

    fn cuda_fwd(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        match q.dtype() {
            candle::DType::F16 => self.cuda_fwd_t::<f16>(q, q_l),
            candle::DType::BF16 => self.cuda_fwd_t::<bf16>(q, q_l),
            candle::DType::F32 => self.cuda_fwd_t::<f32>(q, q_l),
            dt => candle::bail!("interpolation 2d is only supported for f16/bf16/f32 ({dt:?})"),
        }
    }
}

/// Interpole un tenseur 2D (ou batch de 2D) vers une résolution cible (target_h, target_w).
pub fn cuda_interpolate_2d(t: &Tensor, target_h: usize, target_w: usize, interp_type: usize) -> Result<Tensor> {
    let op = SInterp2D {target_h, target_w, interp_type};
    let output = t.apply_op1(op)?;
    Ok(output)
}



pub struct AttnCUDA {
    cuda_attn_type : i32 }

impl AttnCUDA {
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        k: &candle::CudaStorage,
        k_l: &Layout,
        v: &candle::CudaStorage,
        v_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {

        let dev = q.device();
        let out_shape = q_l.shape().clone();

        let q = q.as_cuda_slice::<T>()?;
        let k = k.as_cuda_slice::<T>()?;
        let v = v.as_cuda_slice::<T>()?;
        let q = q.slice(q_l.start_offset()..);
        let k = k.slice(k_l.start_offset()..);
        let v = v.slice(v_l.start_offset()..);

        // Check of Input shapes ..
        let q_stride = q_l.stride();
        let k_stride = k_l.stride();
        let v_stride = v_l.stride();

        let q_rank = q_stride.len();
        let k_rank = k_stride.len();
        let v_rank = v_stride.len();

        if q_rank <= 2 || k_rank <= 2 || v_rank <= 2 {
            candle::bail!(
                "flash-attn expects input tensors of rank above 2 (q: {q_rank}, k: {k_rank}, v: {v_rank}"
            )
        }
        if q_stride[q_rank - 1] != 1 {
            candle::bail!("the last dim of q must be contiguous {q_stride:?}")
        }
        if k_stride[k_rank - 1] != 1 {
            candle::bail!("the last dim of k must be contiguous {k_stride:?}")
        }
        if v_stride[v_rank - 1] != 1 {
            candle::bail!("the last dim of v must be contiguous {v_stride:?}")
        }

        if (q_l.shape() != k_l.shape()) || (q_l.shape() != v_l.shape()) {
            candle::bail!("shape mismatch q {:?}, k {:?}, v {:?}", q_l.shape(), k_l.shape(), v_l.shape())
        }

        let elem_count = out_shape.elem_count();
        //println!("Debug:{}", elem_count);
        let dst = unsafe { dev.alloc::<T>(elem_count) }?;

        let dims: Vec<i32> = q_l.dims().into_iter().map(|&x| x as i32).collect();

        let n : i32 = dims[dims.len() - 2];   // Transformer tokens nb  (e.g. 1374 for DINOv2Reg4: 518/14 * 518/14 + 1cls + 4regs)
        let d : i32 = dims[dims.len() - 1];   // Transformer hidden dim (e.g. 768 for Base)
        let b : i32 = dims.iter().rev().skip(2).rev().fold(1, |acc, x| acc * x); // batch including 12 heads

        let stream = dev.cuda_stream(); 
        unsafe {
            let (q_ptr, _guard) = q.device_ptr(&stream);
            let (k_ptr, _guard) = k.device_ptr(&stream);
            let (v_ptr, _guard) = v.device_ptr(&stream);
            let (dst_ptr, _guard) = dst.device_ptr(&stream);
            ffi::ffi_attn_cuda(
                q_ptr as *const core::ffi::c_void,
                k_ptr as *const core::ffi::c_void,
                v_ptr as *const core::ffi::c_void,
                dst_ptr as *const core::ffi::c_void,
                n,
                d,
                b,
                self.cuda_attn_type,
            )
        }

        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, out_shape))
    }
}

impl candle::CustomOp3 for AttnCUDA {
    fn name(&self) -> &'static str {
        "attn"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for attn")
    }

    fn cuda_fwd(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        k: &candle::CudaStorage,
        k_l: &Layout,
        v: &candle::CudaStorage,
        v_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        match q.dtype() {
            candle::DType::F16 => self.cuda_fwd_t::<f16>(q, q_l, k, k_l, v, v_l),
            candle::DType::BF16 => self.cuda_fwd_t::<bf16>(q, q_l, k, k_l, v, v_l),
            candle::DType::F32 => self.cuda_fwd_t::<f32>(q, q_l, k, k_l, v, v_l),
            dt => candle::bail!("attn is only supported for f16/bf16/f32 ({dt:?})"),
        }
    }
}

/// Calcule l'attention (QKV) avec un kernel CUDA optimisé.
/// - `cuda_attn_type` : Type d'attention 
pub fn cuda_attn_generic(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cuda_attn_type: i32,
) -> Result<Tensor> {

    let op = AttnCUDA {cuda_attn_type};
    let output = q.apply_op3(k, v, op)?;
    Ok(output)
}

