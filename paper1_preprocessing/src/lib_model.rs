//! Module for loading and encapsulating **DINOv2** models for **PlantCLEF 2025**
//!
//! **Features**:
//! - Load **ViT-Small** and **ViT-Base** (PC24) models from Hugging Face.
//! - Support for **DINOv2 Window Shifted Attention (WSA)**.
//! - `FeaturesModule` and `HeadModule` traits for inferring features or predictions.
//! - Backend: `candle`.

use candle_core::{D, DType, Module, Result as cResult, Tensor};
use candle_nn::VarBuilder;

use crate::dinov2reg4_mod as dinov2;
use crate::dinov2reg4_wsa as dinov2_wsa;

/// Function to load the DINOv2 small model from Candle.
pub fn load_model_dinov2_small(
    device: &candle_core::Device,
) -> cResult<dinov2::DinoVisionTransformer> {
    println!("Appel à load_model_dinov2_small()");
    let api = hf_hub::api::sync::Api::new().expect("Erreur dans hf hub api sync");
    let api = api.model("lmz/candle-dino-v2".into());
    let model_file = api
        .get("dinov2_vits14.safetensors")
        .expect("Erreur dans le chargement du modèle depuis hugging face.");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, device)? };
    let model = dinov2::vit_small(vb)?;
    println!("model built");
    Ok(model)
}

/// Function to load the DINOv2 base PlantCLEF2024 model.
pub fn load_model_dinov2_base_pc24(
    device: &candle_core::Device,
) -> cResult<dinov2::DinoVisionTransformer> {
    println!("Appel à load_model_dinov2_base_pc24()");
    let api = hf_hub::api::sync::Api::new().expect("Erreur dans hf hub api sync");
    let api = api.model("vincent-espitalier/dino-v2-reg4-with-plantclef2024-weights".into());
    let model_file = api
        .get("vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all.safetensors")
        .expect("Erreur dans le chargement du modèle depuis hugging face.");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, device)? };
    let model = dinov2::vit_base(vb)?;
    println!("model built");
    Ok(model)
}

/// Function to load the DINOv2 base PC24 model in WSA (Window Shifted Attention) mode.
pub fn load_model_dinov2_base_pc24_wsa(
    device: &candle_core::Device,
    window_size: usize,
    global_attn_indexes: &[usize],
    attn_cuda_type: usize,
) -> cResult<dinov2_wsa::DinoVisionTransformerWSA> {
    println!("Appel à load_model_dinov2_base_pc24_wsa()");
    let api = hf_hub::api::sync::Api::new().expect("Erreur dans hf hub api sync");
    let api = api.model("vincent-espitalier/dino-v2-reg4-with-plantclef2024-weights".into());
    let model_file = api
        .get("vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all.safetensors")
        .expect("Erreur dans le chargement du modèle depuis hugging face.");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, device)? };
    let model = dinov2_wsa::vit_base(vb, window_size, global_attn_indexes, attn_cuda_type)?;
    println!("model built");
    Ok(model)
}

pub fn load_model_dinov2(
    b_use_pc24model: bool,
    device: &candle_core::Device,
) -> cResult<dinov2::DinoVisionTransformer> {
    Ok(if b_use_pc24model {
        load_model_dinov2_base_pc24(device)?
    } else {
        load_model_dinov2_small(device)?
    })
}

/// Traits for deep features computation
pub trait FeaturesModule {
    fn forward_features(&self, xs: &Tensor) -> cResult<Tensor>;
}

pub struct FeaturesModel<'a, T: FeaturesModule + ?Sized> {
    pub features_module: &'a T,
}

impl<'a, T: FeaturesModule + ?Sized> Module for FeaturesModel<'a, T> {
    fn forward(&self, xs: &Tensor) -> cResult<Tensor> {
        self.features_module.forward_features(xs)
    }
}

/// Traits for head computation (linear classifier)
pub trait HeadModule {
    fn forward_head(&self, xs: &Tensor) -> cResult<Tensor>;
}

#[allow(dead_code)]
pub struct HeadModel<'a, T: HeadModule + ?Sized> {
    pub head_module: &'a T,
}

impl<'a, T: HeadModule + ?Sized> Module for HeadModel<'a, T> {
    fn forward(&self, xs: &Tensor) -> cResult<Tensor> {
        self.head_module.forward_head(xs)
    }
}

/// Struct for head computation (linear classifier) for tiling
pub struct TilingHeadModel<'a, T: HeadModule + ?Sized> {
    pub head_model: &'a T,
}

impl<'a, T: HeadModule + ?Sized> Module for TilingHeadModel<'a, T> {
    fn forward(&self, xs: &Tensor) -> cResult<Tensor> {
        let (b, _n, _h) = xs
            .dims3()
            .expect("Erreur: 3 dimensions attendues (B, N, H)");
        assert_eq!(b, 1, "Le batch size doit être égal à 1");
        //let res = self.model.forward(xs)?.max(1)?;
        let res = candle_nn::ops::softmax(&self.head_model.forward_head(xs)?, D::Minus1)?.max(1)?;
        let dims = res.dims();
        println!("res dims: {dims:?}");
        Ok(res)
    }
}
