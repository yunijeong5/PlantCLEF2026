//! Module implementing core structures for ML data loading management
//!
//! This module provides traits (`Dataset`, `Dataloader`, `collate`), similar to Python's,
//! along with their implementations for **PlantCLEF 2025** use cases.
//!
//! ### Available Structures:
//! - **`SimpleDataset`**: Loads images from a directory.
//! - **`WandTilesDataset`**: Serves image tiles as tensors
//!   *Backend*: Wand/ImageMagick for loading and resizing.
//! - **`TensorDataset`**: Loads precomputed tensors from a directory (for *deep features*).

use std::error::Error;
use std::fmt::Debug;
use std::io;
use std::path::{Path, PathBuf};

use image::{DynamicImage, ImageReader};

use candle_core::{Device, Tensor};

use magick_rust::MagickWand;
use std::io::Cursor;

use crate::lib_files::{
    ImgVecU8, get_folder_images_list, get_folder_tensors_list, load_image,
    load_image_and_resize_with_wand3, load_tensor,
};

pub const IMAGENET_MEAN: [f32; 3] = [0.485f32, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229f32, 0.224, 0.225];

/// Trait for image transformations (applied after loading).
pub trait Transform<I, O> {
    fn apply(&self, input: &I) -> Result<O, Box<dyn Error>>;
}

pub struct ImgToTensorResizeImageNetNormTransform {
    pub resize_h: Option<usize>,
    pub resize_w: Option<usize>,
    pub device: Device,
}

/// Implementation of `Transform` to convert images to tensors, with resize and ImageNet normalization (mean, std).
/// Output shape: (B, C, H, W), where B = 1.
impl Transform<ImgVecU8, Tensor> for ImgToTensorResizeImageNetNormTransform {
    fn apply(&self, input: &ImgVecU8) -> Result<Tensor, Box<dyn Error>> {
        let input_data = input.data.clone();
        let mut data = Tensor::from_vec(input_data, (input.height, input.width, 3), &self.device)?
            .permute((2, 0, 1))?;

        if let (Some(target_h), Some(target_w)) = (self.resize_h, self.resize_w) {
            // On resize si les dimensions sont différentes
            if (target_h != input.height) || (target_w != input.width) {
                data = data
                    .unsqueeze(0)?
                    .interpolate2d(target_h, target_w)?
                    .squeeze(0)?; // 'Nearest' type interpolation
            }
        }

        let mean = Tensor::new(&IMAGENET_MEAN, &self.device)?.reshape((3, 1, 1))?;
        let std = Tensor::new(&IMAGENET_STD, &self.device)?.reshape((3, 1, 1))?;
        Ok((data.to_dtype(candle_core::DType::F32)? / 255.)?
            .broadcast_sub(&mean)?
            .broadcast_div(&std)?
            .unsqueeze(0)?)
    }
}

// Implémentation de Transform pour Transformation inverse: tenseurs -> imgs
#[allow(dead_code)]
pub struct TensorToImgWithImageNetDenormTransform {}

impl Transform<Tensor, ImgVecU8> for TensorToImgWithImageNetDenormTransform {
    fn apply(&self, input: &Tensor) -> Result<ImgVecU8, Box<dyn Error>> {
        let (batch, channels, height, width) = input
            .dims4()
            .expect("Le tenseur doit avoir 4 dimensions (B, C, H, W)");
        assert_eq!(batch, 1, "batch must be one.");
        assert_eq!(channels, 3, "Tensor must have 3 channels (RGB)");

        // 1) Dénormalization
        let mut mean_inv = Vec::new();
        let mut std_inv = Vec::new();
        for c in 0..3 {
            //mean_inv.push(-1.0 * IMAGENET_MEAN[c] / IMAGENET_STD[c]);
            mean_inv.push(-IMAGENET_MEAN[c] / IMAGENET_STD[c]);
            std_inv.push(1.0 / &IMAGENET_STD[c]);
        }

        let device = input.device();
        let mean_inv = Tensor::new(mean_inv, device)?.reshape((3, 1, 1))?;
        let std_inv = Tensor::new(std_inv, device)?.reshape((3, 1, 1))?;
        let output = input
            .squeeze(0)?
            .broadcast_sub(&mean_inv)?
            .broadcast_div(&std_inv)?;

        // 2) Construction de la struct de sortie ImgVecU8

        let input: Vec<f32> = output
            .flatten(0, 2)
            .expect("Erreur")
            .to_vec1::<f32>()
            .unwrap();
        //let mut data = Vec::with_capacity((width * height * 3) as usize);
        let mut data = Vec::with_capacity(width * height * 3);

        for y in 0..height {
            for x in 0..width {
                let r = (input[y * width + x] * 255.0).round() as u8;
                let g = (input[height * width + y * width + x] * 255.0).round() as u8;
                let b = (input[2 * height * width + y * width + x] * 255.0).round() as u8;
                data.push(r);
                data.push(g);
                data.push(b);
            }
        }

        Ok(ImgVecU8 {
            data,
            height,
            width,
        })
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////
// Implémentations sur les structs de Dataset

/// Generic trait for dataset operations.
pub trait Dataset<O> {
    fn get_item(&self, index: usize) -> Result<O, Box<dyn Error>>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Dataset implementation serving images from a directory.
pub struct SimpleDataset<P> {
    imgs_list: Vec<PathBuf>,
    transform: Box<dyn Transform<ImgVecU8, P>>,
}

impl<P> SimpleDataset<P> {
    pub fn new(dataset_folder: &Path, transform: Box<dyn Transform<ImgVecU8, P>>) -> Self {
        SimpleDataset {
            imgs_list: get_folder_images_list(dataset_folder)
                .expect("Error in get_folder_images_list ")
                .collect(),
            transform,
        }
    }
}

impl<P> Dataset<(PathBuf, P)> for SimpleDataset<P> {
    fn get_item(&self, index: usize) -> Result<(PathBuf, P), Box<dyn Error>> {
        let p: &PathBuf = &self.imgs_list[index];
        let b_use_crate_image_for_loading = false;
        let img = if b_use_crate_image_for_loading {
            println!("Warning: Chargement de l'image sans Wand; avec la crate image.");
            load_image(p).expect("Error loading img")
        } else {
            //let res = 518;
            load_image_and_resize_with_wand3(
                p.to_str().expect("Error converting path to string"),
                518,
            )
            .expect("Error loading img")
        };
        Ok((p.clone(), self.transform.apply(&img)?))
    }

    fn len(&self) -> usize {
        self.imgs_list.len()
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////
// Implementations for `collate_fn()` functions.
// Trait `Collatable` to enable batching, i.e., gathering items (String, Tensor, etc.).

/// Define the `Collated` enum with Box for recursive variants.
pub enum Collated {
    VecUSize(Vec<usize>),
    VecString(Vec<String>),
    VecPathBuf(Vec<PathBuf>),
    Tensor(Tensor),
    Tuple2(Box<(Collated, Collated)>),
}

// Implémenter Debug pour Collated pour faciliter l'affichage
impl Debug for Collated {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Collated::VecUSize(v) => write!(f, "VecUSize({v:?})"),
            Collated::VecString(v) => write!(f, "VecString({v:?})"),
            Collated::VecPathBuf(v) => write!(f, "VecPathBuf({v:?})"),
            Collated::Tensor(t) => write!(f, "Tensor({t:?})"),
            Collated::Tuple2(t) => write!(f, "Tuple2({t:?})"),
        }
    }
}

// Définir le trait Collatable avec un type associé Output
pub trait Collatable: Sized {
    type Output: Into<Collated>;
    fn collate_fn(v: Vec<Self>) -> Self::Output;
}

// Implémenter Collatable pour usize
impl Collatable for usize {
    type Output = Collated;

    fn collate_fn(v: Vec<usize>) -> Self::Output {
        Collated::VecUSize(v)
    }
}

// Implémenter Collatable pour String
impl Collatable for String {
    type Output = Collated;

    fn collate_fn(v: Vec<String>) -> Self::Output {
        Collated::VecString(v)
    }
}

// Implémenter Collatable pour PathBuf
impl Collatable for PathBuf {
    type Output = Collated;

    fn collate_fn(v: Vec<PathBuf>) -> Self::Output {
        Collated::VecPathBuf(v)
    }
}

// Implémenter Collatable pour Tensor
impl Collatable for Tensor {
    type Output = Collated;

    fn collate_fn(v: Vec<Tensor>) -> Self::Output {
        if v.is_empty() {
            panic!(
                "Erreur dans collate_fn(): Le vecteur de tenseurs est vide, ce qui n'est pas autorisé."
            );
        }

        //let concatenated = Tensor::stack(&v, 0).unwrap();
        let concatenated = Tensor::cat(&v, 0).unwrap();
        Collated::Tensor(concatenated)
    }
}

// Implémenter Collatable pour un tuple de deux éléments
impl<A, B> Collatable for (A, B)
where
    A: Collatable<Output = Collated>,
    B: Collatable<Output = Collated>,
{
    type Output = Collated;

    fn collate_fn(v: Vec<(A, B)>) -> Self::Output {
        let (a_vec, b_vec): (Vec<_>, Vec<_>) = v.into_iter().unzip();
        let collated_a = A::collate_fn(a_vec);
        let collated_b = B::collate_fn(b_vec);
        Collated::Tuple2(Box::new((collated_a, collated_b)))
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////
// Implementations for `Dataloader` structs.

/// Generic trait for dataloader iteration.
pub trait Dataloader: Iterator {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// Implémentation de Dataloader servant les items un par un (pas de batching)
pub struct SimpleIteratorDataloader<O> {
    dataset: Box<dyn Dataset<O>>,
    index: usize,
}

#[allow(dead_code)]
impl<O> SimpleIteratorDataloader<O>
where
    O: Collatable,
{
    pub fn new(dataset: Box<dyn Dataset<O>>) -> Self {
        SimpleIteratorDataloader { dataset, index: 0 }
    }
}

impl<O> Dataloader for SimpleIteratorDataloader<O>
where
    O: Collatable,
{
    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl<O> Iterator for SimpleIteratorDataloader<O>
where
    O: Collatable,
{
    type Item = O::Output;

    fn next(&mut self) -> Option<O::Output> {
        if self.index >= self.len() {
            return None;
        }
        let output = vec![
            self.dataset
                .get_item(self.index)
                .expect("Error in getting item"),
        ];
        self.index += 1;
        Some(O::collate_fn(output))
    }
}

// Implémentation de Dataloader servant des batchs d'items
pub struct BatchIteratorDataloader<O>
where
    O: Collatable,
{
    dataset: Box<dyn Dataset<O>>,
    batch_size: usize,
    index: usize,
}

impl<O> BatchIteratorDataloader<O>
where
    O: Collatable,
{
    pub fn new(dataset: Box<dyn Dataset<O>>, batch_size: usize) -> Self {
        BatchIteratorDataloader {
            dataset,
            batch_size,
            index: 0,
        }
    }
}

impl<O> Dataloader for BatchIteratorDataloader<O>
where
    O: Collatable,
{
    fn len(&self) -> usize {
        self.dataset.len().div_ceil(self.batch_size)
    }
}

impl<O> Iterator for BatchIteratorDataloader<O>
where
    O: Collatable,
{
    type Item = O::Output;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.dataset.len() {
            return None;
        }

        let end_index = (self.index + self.batch_size).min(self.dataset.len());
        let curr_batch_size = end_index - self.index;
        let mut batch: Vec<O> = Vec::with_capacity(curr_batch_size);

        for i in 0..curr_batch_size {
            batch.push(
                self.dataset
                    .get_item(self.index + i)
                    .expect("Error in getting item"),
            );
        }

        self.index += curr_batch_size;
        Some(O::collate_fn(batch))
    }
}

/// Implementation of `Dataloader` for inference tasks.
/// This is a specialization/interface of `BatchDataloader`, serving only (string, tensor) pairs.
pub struct InferenceDataloader<O>
where
    O: Collatable<Output = Collated>,
{
    loader: BatchIteratorDataloader<O>,
}

impl<O> InferenceDataloader<O>
where
    O: Collatable<Output = Collated>,
{
    pub fn new(dataset: Box<dyn Dataset<O>>, batch_size: usize) -> Self {
        InferenceDataloader {
            loader: BatchIteratorDataloader::new(dataset, batch_size),
        }
    }
}

impl<O> Dataloader for InferenceDataloader<O>
where
    O: Collatable<Output = Collated>,
{
    fn len(&self) -> usize {
        self.loader.len()
    }
}

impl<O> Iterator for InferenceDataloader<O>
where
    O: Collatable<Output = Collated>,
{
    type Item = (Vec<String>, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        let coll = self.loader.next();
        match coll {
            None => None,

            Some(Collated::Tuple2(boxed_tuple)) => {
                let (collated_a, collated_b) = *boxed_tuple;
                match (collated_a, collated_b) {
                    (Collated::VecString(ids), Collated::Tensor(input)) => Some((ids, input)),

                    (Collated::VecUSize(ids), Collated::Tensor(input)) => {
                        Some((ids.iter().map(|&n| n.to_string()).collect(), input))
                    }

                    (Collated::VecPathBuf(ids), Collated::Tensor(input)) => Some((
                        ids.iter()
                            .map(|p| p.to_string_lossy().into_owned())
                            .collect(),
                        input,
                    )),

                    _ => {
                        panic!("Expected Tuple2 to contain VecString and Tensor");
                    }
                }
            }
            _ => {
                panic!(
                    "Expected a Tuple2. Make sure the Dataset Struct provides ( _ , Tensor) Tuple2."
                );
            }
        }
    }
}

// Code for image tiling.

/// Implementation serving tiles from a tensor representing an image.
#[derive(Debug, Clone, Copy)]
pub struct Geometry {
    pub height: usize,
    pub width: usize,
    pub height_offset: isize,
    pub width_offset: isize,
}

/// Dataset for loading tiles using Wand.
pub struct WandTilesDataset {
    img_wand: MagickWand,
    geo: Vec<Geometry>,
    device: Device,
}

#[allow(clippy::ptr_arg)]
impl WandTilesDataset {
    pub fn init_with_geometry(
        img_path: &str,
        res_base: usize,
        geo: &Vec<Geometry>,
        device: &Device,
    ) -> Result<Self, Box<dyn Error>> {
        let wand = MagickWand::new();
        wand.read_image(img_path)?;
        let old_height = wand.get_image_height();
        let old_width = wand.get_image_width();

        let (height, width): (usize, usize) = if old_height < old_width {
            (
                res_base,
                ((res_base as f32) * (old_width as f32) / (old_height as f32)).round() as usize,
            )
        } else {
            (
                ((res_base as f32) * (old_height as f32) / (old_width as f32)).round() as usize,
                res_base,
            )
        };

        let (crop_height, crop_width): (isize, isize) = if old_height < old_width {
            (0, ((width - res_base) / 2).try_into().unwrap())
        } else {
            (((height - res_base) / 2).try_into().unwrap(), 0)
        };

        wand.resize_image(width, height, magick_rust::FilterType::Lanczos)?;
        //let height = wand.get_image_height();
        //let width = wand.get_image_width();

        wand.crop_image(res_base, res_base, crop_width, crop_height)
            .expect("Error while crop");
        let height = wand.get_image_height();
        let width = wand.get_image_width();
        if (height != res_base) || (width != res_base) {
            let err_msg = format!(
                "Erreur dans le calcul des dimensions d'image:{},{},{},{},{},{}",
                old_height, old_width, height, width, crop_height, crop_width
            );
            return Err(Box::new(io::Error::other(err_msg)));
        }

        // reset_image_page() = reset_coords() en Python = "+repage" en CLI
        wand.reset_image_page("")
            .expect("Error in reset_image_page");

        Ok(WandTilesDataset {
            img_wand: wand,
            geo: geo.clone(),
            device: device.clone(),
        })
    }
}

impl Dataset<(usize, Tensor)> for WandTilesDataset {
    fn len(&self) -> usize {
        self.geo.len()
    }

    fn get_item(&self, index: usize) -> Result<(usize, Tensor), Box<dyn Error>> {
        let tile_index = index; // For clarity

        // Build the Tensor from the cropped wand image..
        let wand = self.img_wand.clone();
        let res_base_height = wand.get_image_height();
        let res_base_width = wand.get_image_width();
        //let (height, width, crop_height, crop_width) = self.geo[index];
        let geo: Geometry = self.geo[index];

        wand.crop_image(geo.width, geo.height, geo.width_offset, geo.height_offset)
            .expect("Error while crop");
        let new_height = wand.get_image_height();
        let new_width = wand.get_image_width();
        if (geo.height != new_height) || (geo.width != new_width) {
            let err_msg = format!(
                "Erreur dans le calcul des dimensions de la tuile {}: res_base {},{};  new_res {},{};  crop_res:{},{};  crop_offsets:{},{}",
                tile_index,
                res_base_height,
                res_base_width,
                new_height,
                new_width,
                geo.height,
                geo.width,
                geo.height_offset,
                geo.width_offset
            );
            return Err(Box::new(io::Error::other(err_msg)));
        }

        let data = wand
            .write_image_blob("jpeg")
            .expect("Error converting to blob");

        let cursor = Cursor::new(data);
        // Chargez l'image à partir des données binaires
        let data = ImageReader::new(cursor)
            .with_guessed_format()
            .expect("Failed to guess image format")
            .decode()
            .expect("Failed to decode image")
            .to_rgb8()
            .into_raw();

        let transform = ImgToTensorResizeImageNetNormTransform {
            resize_h: None,
            resize_w: None,
            device: self.device.clone(),
        };

        Ok((
            tile_index,
            transform.apply(&ImgVecU8 {
                data,
                height: geo.height,
                width: geo.width,
            })?,
        ))
    }
}

/// Implementation serving tiles from a tensor representing an image.
/// Loading using the `image` crate (Rust's standard library).
pub struct TilesDataset {
    img: DynamicImage,
    geo: Vec<Geometry>,
    device: Device,
}

#[allow(clippy::ptr_arg)]
impl TilesDataset {
    pub fn init_with_geometry(
        img_path: &str,
        res_base: usize,
        geo: &Vec<Geometry>,
        device: &Device,
    ) -> Result<Self, Box<dyn Error>> {
        let img = image::ImageReader::open(img_path)?;
        let img = img.decode()?;

        let old_height = img.height();
        let old_width = img.width();

        let (height, width): (usize, usize) = if old_height < old_width {
            (
                res_base,
                ((res_base as f32) * (old_width as f32) / (old_height as f32)).round() as usize,
            )
        } else {
            (
                ((res_base as f32) * (old_height as f32) / (old_width as f32)).round() as usize,
                res_base,
            )
        };

        let (crop_height, crop_width): (isize, isize) = if old_height < old_width {
            (0, ((width - res_base) / 2).try_into().unwrap())
        } else {
            (((height - res_base) / 2).try_into().unwrap(), 0)
        };

        let img = img.resize_to_fill(
            width as u32,
            height as u32,
            image::imageops::FilterType::Lanczos3,
        );

        //let height = img.height();
        //let width = img.width();

        let img = img.crop_imm(
            crop_width as u32,
            crop_height as u32,
            res_base as u32,
            res_base as u32,
        );
        let height = img.height();
        let width = img.width();
        if (height != res_base as u32) || (width != res_base as u32) {
            let err_msg = format!(
                "Erreur dans le calcul des dimensions d'image:{},{},{},{},{},{}",
                old_height, old_width, height, width, crop_height, crop_width
            );
            return Err(Box::new(io::Error::other(err_msg)));
        }

        Ok(TilesDataset {
            img: img.clone(),
            geo: geo.clone(),
            device: device.clone(),
        })
    }
}

impl Dataset<(usize, Tensor)> for TilesDataset {
    fn len(&self) -> usize {
        self.geo.len()
    }

    fn get_item(&self, index: usize) -> Result<(usize, Tensor), Box<dyn Error>> {
        let tile_index = index; // For clarity

        // Build the Tensor from the cropped wand image..
        let img = self.img.clone();
        let res_base_height = img.height();
        let res_base_width = img.width();
        let geo: Geometry = self.geo[index];

        let img = img.crop_imm(
            geo.width_offset as u32,
            geo.height_offset as u32,
            geo.width as u32,
            geo.height as u32,
        );
        let new_height = img.height();
        let new_width = img.width();
        if (geo.height as u32 != new_height) || (geo.width as u32 != new_width) {
            let err_msg = format!(
                "Erreur dans le calcul des dimensions de la tuile {}: res_base {},{};  new_res {},{};  crop_res:{},{};  crop_offsets:{},{}",
                tile_index,
                res_base_height,
                res_base_width,
                new_height,
                new_width,
                geo.height,
                geo.width,
                geo.height_offset,
                geo.width_offset
            );
            return Err(Box::new(io::Error::other(err_msg)));
        }

        let img = img.to_rgb8();
        let (width, height) = img.dimensions();
        let data = img.into_raw();

        let transform = ImgToTensorResizeImageNetNormTransform {
            resize_h: None,
            resize_w: None,
            device: self.device.clone(),
        };

        Ok((
            tile_index,
            transform.apply(&ImgVecU8 {
                data,
                height: height as usize,
                width: width as usize,
            })?,
        ))
    }
}

/// Implementation serving safetensors files from a directory on disk.
pub struct TensorDataset {
    tensors_list: Vec<PathBuf>,
    device: Device,
}

impl TensorDataset {
    pub fn new(dataset_folder: &Path, device: Device) -> Self {
        TensorDataset {
            tensors_list: get_folder_tensors_list(dataset_folder)
                .expect("Error in get_folder_tensors_list")
                .collect(),
            device,
        }
    }
}

impl Dataset<(PathBuf, Tensor)> for TensorDataset {
    fn get_item(&self, index: usize) -> Result<(PathBuf, Tensor), Box<dyn Error>> {
        let p: &PathBuf = &self.tensors_list[index];
        let tensor = load_tensor(p, &self.device)
            .expect("Error loading img")
            .unsqueeze(0)?;
        Ok((p.clone(), tensor))
    }

    fn len(&self) -> usize {
        self.tensors_list.len()
    }
}
