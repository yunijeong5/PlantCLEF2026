//! Implementation of the DINOv2 model with 4 registers and Window Sliding Attention (WSA)
//!
//! Adapted version from Candle crate
//! Following SegmentAnything form: Mix of WSA and global attentions, specificied by the global_attn_index list.
//!
//! The DINOv2-reg4 model is a variant of DINOv2 that adds 4 regularization tokens to the
//! original architecture. This implementation is specifically trained for plant species
//! classification on the PlantCLEF2024 dataset with 7,806 classes.
//!
//! - [Paper](https://arxiv.org/abs/2309.16588). DINOv2: Learning Robust Visual Features without Supervision
//! - [GH Repo](https://github.com/facebookresearch/dinov2)
//!
use candle_core::{D, Device, IndexOp, Result, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder, layer_norm};

use crate::lib_model::{FeaturesModule, HeadModule};
//use lib_cuda_rs::cuda_interpolate_2d;

const IMG_SIZE: usize = 518;
const PATCH_SIZE: usize = 14;
const NUM_CLASSES: usize = 7806; // PlantCLEF2024 DINOv2 (https://zenodo.org/records/10848263)

// MistralAI LeChat proposition for bicubic interpolation over Candle Tensors:
#[allow(dead_code)]
fn cubic_interpolation(x: f32) -> f32 {
    let abs_x = x.abs();
    if abs_x <= 1.0 {
        1.0 - 2.0 * abs_x * abs_x + abs_x * abs_x * abs_x
    } else if abs_x < 2.0 {
        4.0 - 8.0 * abs_x + 5.0 * abs_x * abs_x - abs_x * abs_x * abs_x
    } else {
        0.0
    }
}

#[allow(dead_code)]
fn interpolate_bicubic_cpu(tensor: &Tensor, new_height: usize, new_width: usize) -> Result<Tensor> {
    let (batch_size, channels, height, width) = tensor.dims4()?;
    let tensor = tensor.to_device(&Device::Cpu)?;

    //let mut output = Tensor::zeros((batch_size, channels, new_height, new_width), tensor.dtype(), tensor.device())?;
    let mut output_data = vec![0.0; batch_size * channels * new_height * new_width];

    let height_ratio = height as f32 / new_height as f32;
    let width_ratio = width as f32 / new_width as f32;

    println!(
        "interpolate_bicubic_cpu: b={batch_size}, c={channels}, h={height}, w={width}, nh={new_height}, nw={new_width}"
    );

    for b in 0..batch_size {
        for c in 0..channels {
            for y in 0..new_height {
                for x in 0..new_width {
                    let src_x = (x as f32 + 0.5) * width_ratio - 0.5;
                    let src_y = (y as f32 + 0.5) * height_ratio - 0.5;

                    let x_floor = src_x.floor() as i32;
                    let y_floor = src_y.floor() as i32;

                    let mut sum = 0.0;
                    for ky in -1..=2 {
                        for kx in -1..=2 {
                            let yy = y_floor + ky;
                            let xx = x_floor + kx;

                            if yy >= 0 && yy < height as i32 && xx >= 0 && xx < width as i32 {
                                let weight = cubic_interpolation(src_x - xx as f32)
                                    * cubic_interpolation(src_y - yy as f32);
                                let value = tensor
                                    .get(b)?
                                    .get(c)?
                                    .get(yy as usize)?
                                    .get(xx as usize)?
                                    .to_scalar::<f32>()?;
                                sum += value * weight;
                            }
                        }
                    }
                    let index = b * channels * new_height * new_width
                        + c * new_height * new_width
                        + y * new_width
                        + x;
                    output_data[index] = sum;
                }
            }
        }
    }

    let output = Tensor::from_vec(
        output_data,
        (batch_size, channels, new_height, new_width),
        &Device::Cpu,
    )?;
    Ok(output)
}

fn linear(vb: VarBuilder, in_dim: usize, out_dim: usize, bias: bool) -> Result<Linear> {
    if bias {
        candle_nn::linear(in_dim, out_dim, vb)
    } else {
        candle_nn::linear_no_bias(in_dim, out_dim, vb)
    }
}

#[derive(Debug)]
struct Attention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    scale: f64,
    attn_cuda_type: usize,
}

//#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    //use candle_flash_attn::flash_attn;
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

/*
#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}
*/

impl Attention {
    fn new(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        proj_bias: bool,
        attn_cuda_type: usize,
    ) -> Result<Self> {
        let qkv = linear(vb.pp("qkv"), dim, dim * 3, qkv_bias)?;
        let proj = linear(vb.pp("proj"), dim, dim, proj_bias)?;
        let scale = 1. / ((dim / num_heads) as f64).sqrt();
        Ok(Self {
            qkv,
            proj,
            num_heads,
            scale,
            attn_cuda_type,
        })
    }
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, n, c) = xs.dims3()?;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((b, n, 3, self.num_heads, c / self.num_heads))?
            .transpose(1, 2)? // 02134
            .transpose(0, 1)? // 20134
            .transpose(2, 3)?; // 20314
        let q = (qkv.i(0)? * self.scale)?;
        let k = qkv.i(1)?.contiguous()?;
        let v = qkv.i(2)?.contiguous()?;

        // Default = 0 => Explicit attn
        let use_f16_attn = self.attn_cuda_type == 1; // Explicit attn in f16
        let use_flash_attn = self.attn_cuda_type == 2; // Flash attn
        let use_cuda_attn = self.attn_cuda_type == 3; // CUDA attn (low memory footprint)

        let attn_output = if use_flash_attn {
            let init_dtype = q.dtype();
            let causal = false;
            let scale = 1.0; // Scale déjà appliqué sur q
            let q = q.to_dtype(candle_core::DType::F16)?.transpose(1, 2)?;
            let k = k.to_dtype(candle_core::DType::F16)?.transpose(1, 2)?;
            let v = v.to_dtype(candle_core::DType::F16)?.transpose(1, 2)?;
            flash_attn(&q, &k, &v, scale, causal)?
                .transpose(1, 2)?
                .to_dtype(init_dtype)?
        } else if use_cuda_attn {
            use lib_cuda_rs::cuda_attn_generic;
            cuda_attn_generic(&q, &k, &v, 3)?
        } else if use_f16_attn {
            let q = q.to_dtype(candle_core::DType::F16)?;
            let k = k.to_dtype(candle_core::DType::F16)?;
            let v = v.to_dtype(candle_core::DType::F16)?;
            candle_nn::ops::softmax(&q.matmul(&k.t()?)?, D::Minus1)?
                .matmul(&v)?
                .to_dtype(candle_core::DType::F32)?
        } else {
            // F32 arithm // Standard implementation
            candle_nn::ops::softmax(&q.matmul(&k.t()?)?, D::Minus1)?.matmul(&v)?
        };

        let attn = attn_output.transpose(1, 2)?.reshape((b, n, c))?;
        self.proj.forward(&attn)
    }
}

#[derive(Debug)]
struct LayerScale {
    gamma: Tensor,
}

impl LayerScale {
    fn new(vb: VarBuilder, dim: usize) -> Result<Self> {
        let gamma = vb.get(dim, "gamma")?;
        Ok(Self { gamma })
    }
}

impl Module for LayerScale {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&self.gamma)
    }
}

#[derive(Debug)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new(vb: VarBuilder, in_features: usize, hidden_features: usize, bias: bool) -> Result<Self> {
        let out_features = in_features;
        let fc1 = linear(vb.pp("fc1"), in_features, hidden_features, bias)?;
        let fc2 = linear(vb.pp("fc2"), hidden_features, out_features, bias)?;
        Ok(Self { fc1, fc2 })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?.gelu()?;
        self.fc2.forward(&xs)
    }
}

#[derive(Debug)]
struct Block {
    norm1: LayerNorm,
    attn: Attention,
    ls1: LayerScale,
    norm2: LayerNorm,
    mlp: Mlp,
    ls2: LayerScale,
    window_size: usize,
}

impl Block {
    fn new(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        window_size: usize,
        attn_cuda_type: usize,
    ) -> Result<Self> {
        let norm1 = layer_norm(dim, 1e-6, vb.pp("norm1"))?;
        let attn = Attention::new(vb.pp("attn"), dim, num_heads, true, true, attn_cuda_type)?;
        let ls1 = LayerScale::new(vb.pp("ls1"), dim)?;
        let norm2 = layer_norm(dim, 1e-6, vb.pp("norm2"))?;
        let mlp = Mlp::new(vb.pp("mlp"), dim, dim * 4, true)?;
        let ls2 = LayerScale::new(vb.pp("ls2"), dim)?;
        Ok(Self {
            norm1,
            attn,
            ls1,
            norm2,
            mlp,
            ls2,
            window_size,
        })
    }
}

fn window_partition(xs: Tensor, window_size: usize) -> Result<(Tensor, usize)> {
    let num_prefix_tokens = 1 + 4; // CLS + 4 regs
    let (b, n, c) = xs.dims3()?;
    let sqrt_n = ((n - num_prefix_tokens) as f64).sqrt() as usize;
    if !sqrt_n.is_multiple_of(window_size) {
        candle_core::bail!(
            "Square of patch nb tokens sqrt(n-5) (n={n}) is not a multiple of the square of window size (ws={window_size})"
        )
    }
    let k = sqrt_n / window_size;

    let n_tuiles = k * k;
    println!("window_partition: n_tuiles={n_tuiles}, b={b}, n={n}, c={c}");

    let prefix_tokens = xs.narrow(1, 0, num_prefix_tokens)?;
    let xs = xs.narrow(1, num_prefix_tokens, n - num_prefix_tokens)?;

    let prefix_tokens = prefix_tokens
        .unsqueeze(1)?
        .expand(&[b, k * k, num_prefix_tokens, c])?
        .reshape((b * k * k, num_prefix_tokens, c))?
        .contiguous()?;

    let windows = xs
        .reshape((b, k, window_size, k, window_size, c))?
        .transpose(2, 3)?
        .reshape((b * k * k, window_size * window_size, c))?
        .contiguous()?;

    let windows = Tensor::cat(&[prefix_tokens, windows], 1)?.contiguous()?;

    Ok((windows, k))
}

fn window_unpartition(windows: Tensor, window_size: usize, k: usize) -> Result<Tensor> {
    let num_prefix_tokens = 1 + 4; // CLS + 4 regs
    let (b2, n, c) = windows.dims3()?;
    if (b2 % (k * k)) != 0 {
        candle_core::bail!(
            "Internal error: nb of windows batch (b2={b2}) is not a multiple of the square of k (k={k})"
        )
    }
    let b = b2 / k / k;

    let prefix_tokens = windows.narrow(1, 0, num_prefix_tokens)?;
    let windows = windows.narrow(1, num_prefix_tokens, n - num_prefix_tokens)?;

    // Aggrégation des tokens globaux (CLS et reg) par moyennage tile-wise
    let prefix_tokens = prefix_tokens
        .reshape((b, k * k, num_prefix_tokens, c))?
        .mean(1)?
        .contiguous()?;

    let xs = windows
        .reshape((b, k, k, window_size, window_size, c))?
        .transpose(2, 3)?
        .reshape((b, k * window_size * k * window_size, c))?
        .contiguous()?;

    let xs = Tensor::cat(&[prefix_tokens, xs], 1)?.contiguous()?;
    Ok(xs)
}

impl Module for Block {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.norm1.forward(xs)?;

        let (xs, k) = if self.window_size > 0 {
            window_partition(xs, self.window_size)?
        } else {
            (xs, 0)
        };

        let xs = self.attn.forward(&xs)?;

        let xs = if self.window_size > 0 {
            window_unpartition(xs, self.window_size, k)?
        } else {
            xs
        };

        let xs = self.ls1.forward(&xs)?;
        let xs = (xs + residual)?;

        let residual = &xs;
        let xs = self
            .ls2
            .forward(&self.mlp.forward(&self.norm2.forward(&xs)?)?)?;
        xs + residual
    }
}

#[derive(Debug)]
struct PatchEmbed {
    proj: candle_nn::Conv2d,
    patch_size: (usize, usize),
    //num_patches: usize,
}

impl PatchEmbed {
    fn new(
        vb: VarBuilder,
        //img_size: usize,
        patch_size: usize,
        in_chans: usize,
        embed_dim: usize,
    ) -> Result<Self> {
        let config = candle_nn::Conv2dConfig {
            stride: patch_size,
            ..Default::default()
        };
        let proj = candle_nn::conv2d(in_chans, embed_dim, patch_size, config, vb.pp("proj"))?;
        //let num_patches = (img_size / patch_size) * (img_size / patch_size);
        Ok(Self {
            proj,
            patch_size: (patch_size, patch_size),
            //num_patches,
        })
    }
}

impl Module for PatchEmbed {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b, _c, h, w) = xs.dims4()?;
        let (patch_h, patch_w) = self.patch_size;
        if (h % patch_h) != 0 {
            candle_core::bail!("image height {h} is not a multiple of patch height {patch_h}")
        }
        if (w % patch_w) != 0 {
            candle_core::bail!("image width {w} is not a multiple of patch width {patch_w}")
        }
        let xs = self.proj.forward(xs)?;
        let (b, c, h, w) = xs.dims4()?;
        // flatten embeddings.
        xs.reshape((b, c, h * w))?.transpose(1, 2)
    }
}

#[derive(Debug)]
pub struct DinoVisionTransformerWSA {
    patch_embed: PatchEmbed,
    cls_token: Tensor,
    reg_token: Tensor,
    pos_embed: Tensor,
    blocks: Vec<Block>,
    norm: LayerNorm,
    head: Linear,
}

impl DinoVisionTransformerWSA {
    pub fn new(
        vb: VarBuilder,
        depth: usize,
        embed_dim: usize,
        num_heads: usize,
        window_size: usize,
        global_attn_indexes: &[usize],
        attn_cuda_type: usize,
    ) -> Result<Self> {
        let patch_embed = PatchEmbed::new(vb.pp("patch_embed"), PATCH_SIZE, 3, embed_dim)?;
        //PatchEmbed::new(vb.pp("patch_embed"), IMG_SIZE, PATCH_SIZE, 3, embed_dim)?;
        let cls_token = vb.get((1, 1, embed_dim), "cls_token")?;
        let reg_token = vb.get((1, 4, embed_dim), "reg_token")?;
        let num_patches = (IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE);
        let pos_embed = vb.get((1, num_patches, embed_dim), "pos_embed")?;
        let head = linear(vb.pp("head"), embed_dim, NUM_CLASSES, true)?;
        let norm = layer_norm(embed_dim, 1e-6, vb.pp("norm"))?;
        let vb_b = vb.pp("blocks");
        let blocks = (0..depth)
            .map(|i| {
                let window_size = if global_attn_indexes.contains(&i) {
                    0
                } else {
                    window_size
                };
                Block::new(
                    vb_b.pp(i.to_string()),
                    embed_dim,
                    num_heads,
                    window_size,
                    attn_cuda_type,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_embed,
            cls_token,
            reg_token,
            pos_embed,
            blocks,
            norm,
            head,
        })
    }

    fn interpolate_pos_encoding(&self, xs: &Tensor, w: usize, h: usize) -> Result<Tensor> {
        let (_b, _n, _h) = xs.dims3()?; // _n: xs ne contient pas le cls ni les regs (concaténés apres)
        let npatch = xs.dim(1)?; // - 1; Pas besoin d'enlever 1 pour le cls token,
        let n = self.pos_embed.dim(1)?; // - 1;  car il est ni dans le xs, ni dans le pos_embed
        let sqrt_n = (n as f64).sqrt();
        if npatch == n && w == h {
            return Ok(self.pos_embed.clone());
        }
        println!("VaMISsage du modèle: Interpolation du positional encoding");
        let patch_pos_embed = &self.pos_embed;
        let dim = xs.dim(D::Minus1)?;
        let (w0, h0) = ((w / PATCH_SIZE) as f64 + 0.1, (h / PATCH_SIZE) as f64 + 0.1);
        let patch_pos_embed = patch_pos_embed
            .reshape((1, sqrt_n as usize, sqrt_n as usize, dim))?
            .transpose(2, 3)?
            .transpose(1, 2)?;

        // This uses bicubic interpolation in the original implementation.
        let patch_pos_embed = patch_pos_embed.upsample_nearest2d(h0 as usize, w0 as usize)?;

        // Very slow
        //let patch_pos_embed = interpolate_bicubic_cpu(&patch_pos_embed, h0 as usize, w0 as usize)?;

        //let interp_type = 1; // Nearest
        //let interp_type = 6;
        //let interp_type = 5; // Bicubic
        //let patch_pos_embed = cuda_interpolate_2d(&patch_pos_embed, h0 as usize, w0 as usize, interp_type)?;

        let el_count = patch_pos_embed.shape().elem_count();
        patch_pos_embed
            .transpose(1, 2)?
            .transpose(2, 3)?
            .reshape((1, el_count / dim, dim))
    }

    fn prepare_tokens_with_mask(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, _nc, w, h) = xs.dims4()?;
        /*
        if (w != IMG_SIZE) || (h != IMG_SIZE) {
            panic!("Error: The input tensor should have the shape: Bx3x518x518.");
        }
        */
        let xs = self.patch_embed.forward(xs)?;
        let (_, t, h2) = xs.dims3()?;
        //let xs = (&xs + &self.interpolate_pos_encoding(&xs, w, h)?)?;
        let xs = (&xs
            + &self
                .interpolate_pos_encoding(&xs, w, h)?
                .expand(&[b, t, h2])?)?;
        let xs = Tensor::cat(
            &[
                &self.cls_token.expand(&[b, 1, h2])?,
                &self.reg_token.expand(&[b, 4, h2])?,
                &xs,
            ],
            1,
        )?;
        Ok(xs)
    }
}

impl FeaturesModule for DinoVisionTransformerWSA {
    // On force clippy à nous laisser faire le return en 2 coups
    // pour expliciter que l'on retourne le cls token
    #[allow(clippy::let_and_return)]
    fn forward_features(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.prepare_tokens_with_mask(xs)?;
        for blk in self.blocks.iter() {
            xs = blk.forward(&xs)?
        }
        let xs = self.norm.forward(&xs)?;
        //let xs_norm_clstoken = xs.i((.., 0))?.contiguous();
        let xs_norm_clstoken = xs.i((.., 0))?.contiguous(); // Fix pour batchs (2025-04-21)
        xs_norm_clstoken
    }
}

impl HeadModule for DinoVisionTransformerWSA {
    fn forward_head(&self, xs_norm_clstoken: &Tensor) -> Result<Tensor> {
        self.head.forward(xs_norm_clstoken)
        //Ok(xs_norm_clstoken.clone())
    }
}

impl Module for DinoVisionTransformerWSA {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs_norm_clstoken = self.forward_features(xs)?;
        self.forward_head(&xs_norm_clstoken)
    }
}

#[allow(dead_code)]
pub fn vit_small(
    vb: VarBuilder,
    window_size: usize,
    global_attn_indexes: &[usize],
    attn_cuda_type: usize,
) -> Result<DinoVisionTransformerWSA> {
    DinoVisionTransformerWSA::new(
        vb,
        12,
        384,
        6,
        window_size,
        global_attn_indexes,
        attn_cuda_type,
    )
}

pub fn vit_base(
    vb: VarBuilder,
    window_size: usize,
    global_attn_indexes: &[usize],
    attn_cuda_type: usize,
) -> Result<DinoVisionTransformerWSA> {
    DinoVisionTransformerWSA::new(
        vb,
        12,
        768,
        12,
        window_size,
        global_attn_indexes,
        attn_cuda_type,
    )
}
