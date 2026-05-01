//! Inference module for **PlantCLEF 2025**: prediction computation and *deep features* extraction from images and tiles.
//!
//! **Main Features**:
//! - Batch inference on GPU/CPU with performance tracking.
//! - Support for **DINOv2** models (tiling head or feature extraction).
//! - **Tiling** and **VaMIS** pipeline management (dynamic resizing, sliding windows).
//! - Saving tensors in `safetensors` format and exporting predictions to CSV.

use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{Device, IndexOp, Result as cResult, Tensor};
use candle_nn::Module;

use crate::lib_dataset::{
    Dataloader, Dataset, Geometry, ImgToTensorResizeImageNetNormTransform, InferenceDataloader,
    SimpleDataset, TensorDataset, TilesDataset, Transform, WandTilesDataset,
};

use crate::lib_files::{
    PredictionsDataframe, get_basename, get_folder_images_list, get_folder_tensors_list,
    load_image, load_image_and_resize_with_wand3, read_text_file, save_tensor,
};

use crate::lib_model::{
    FeaturesModel, FeaturesModule, TilingHeadModel, load_model_dinov2,
    load_model_dinov2_base_pc24_wsa,
};

pub fn seconds_since_epoch() -> f64 {
    let now = SystemTime::now();
    let since_the_epoch = now.duration_since(UNIX_EPOCH).expect("Time went backwards");
    since_the_epoch.as_secs() as f64 + since_the_epoch.subsec_nanos() as f64 * 1e-9
}

pub fn get_compute_device(cpu: bool) -> cResult<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

/// Function to perform inferences.
/// Input: an iterator over (strings, tensors) + a model + batch size + device (CPU, GPU).
/// Output: a large tensor.
pub fn calc_inferences_from_iterator(
    dataloader: impl Iterator<Item = (Vec<String>, Tensor)>,
    model: &dyn Module,
    n_batchs: usize,
    n_items: usize,
    verbose: bool,
    storage_device: &Device,
    compute_device: &Device,
) -> cResult<(Vec<String>, Tensor)> {
    if verbose {
        println!("Appel à lib_inference::calc_inferences_from_iterator()");
    }
    let mut output_ids = Vec::new();
    let mut output_tensor =
        Tensor::zeros(&[1, 1], candle_core::DType::F32, compute_device).expect("Error");
    let mut output_tensor_index = 0;
    let t_start = seconds_since_epoch();
    let mut duration_only_inferences = 0.0;
    for (batch_index, (ids, input)) in dataloader.enumerate() {
        let input = input.to_device(compute_device)?.contiguous()?;
        if verbose {
            println!("Input shape:{:?}", input.shape());
        }
        let t_before_inference = seconds_since_epoch();
        let output = model.forward(&input)?; // Modèle inférence
        //let output = input.i((.., 0, 0, ..))?;  // Tester sans réelle inférence (évaluer la durée du chargement)
        //let output = input.i((.., 0, ..))?;  // Tester sans réelle inférence  - tete de tiling
        if verbose {
            println!("Output shape:{:?}", output.shape());
        }
        let output = output.to_device(storage_device)?;
        let t_after_inference = seconds_since_epoch();
        duration_only_inferences += t_after_inference - t_before_inference;
        let duration = t_after_inference - t_start;
        let nb_seconds_per_batch = duration / (batch_index as f64 + 1.);
        let nb_batchs_per_second = 1. / nb_seconds_per_batch;
        let remaining_time = ((n_batchs - (batch_index + 1)) as f64) / nb_batchs_per_second;
        let batch_index_inc = batch_index + 1;
        if verbose {
            println!(
                "Batch {batch_index_inc}/{n_batchs},  remaining time: {remaining_time:.2}s,  nb batch/s: {nb_batchs_per_second:.2},  nb s/batch: {nb_seconds_per_batch:.2},  ids:{ids:?}"
            );
        } else {
            println!("Batch {batch_index_inc}/{n_batchs}");
        }
        std::io::stdout().flush().unwrap();

        output_ids.extend(ids);

        let (bs_cour, c) = output
            .dims2()
            .expect("Erreur dans calc_inferences_from_iterator: 2 dimensions attendues");
        if output_tensor_index == 0 {
            // Allocation
            let shape = &[n_items, c];
            output_tensor =
                Tensor::zeros(shape, candle_core::DType::F32, storage_device).expect("Error");
        }
        output_tensor = output_tensor
            .slice_assign(
                &[output_tensor_index..(output_tensor_index + bs_cour), 0..c],
                &output,
            )
            .expect("N'a pu faire slice_assign");
        output_tensor_index += bs_cour;
    }

    let duration = seconds_since_epoch() - t_start;
    let nb_seconds_per_img = duration / output_ids.len() as f64;
    let nb_imgs_per_second = 1. / nb_seconds_per_img;
    let ratio_inference_duration_pcts = duration_only_inferences / duration * 100.;
    let duration_per_img_outside_inference = (duration - duration_only_inferences) / n_items as f64;
    if verbose {
        println!(" ");
        println!("Inference: Timing statistiques:");
        let len = output_ids.len();
        println!("Total duration: {duration:.2}s for {len} images");
        println!(
            "Pure inference duration: {duration_only_inferences:.2}s ({ratio_inference_duration_pcts:.2}% total), Avg image loading: {duration_per_img_outside_inference:.4}s"
        );
        println!("nb imgs/s: {nb_imgs_per_second:.2}, nb s/img: {nb_seconds_per_img:.2}");
    } else {
        println!("; Total duration: {duration:.2}s");
    }

    Ok((output_ids, output_tensor))
}

/// Iterator implementation for tensors.
/// Allows iterating over the first dimension of a tensor.
pub struct TensorIterator {
    tensor: Tensor,
    index: usize,
}

impl TensorIterator {
    pub fn new(tensor: Tensor) -> Self {
        TensorIterator { tensor, index: 0 }
    }
    pub fn len(&self) -> usize {
        self.tensor.dims()[0]
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Iterator for TensorIterator {
    type Item = Tensor;
    fn next(&mut self) -> Option<Tensor> {
        if self.index >= self.len() {
            return None;
        }
        let t = self
            .tensor
            .i((self.index, ..))
            .expect("N'a pas pu indexer le tenseur");
        self.index += 1;
        Some(t)
    }
}

/// Main inference function (resizes images, not the model).
///
/// # Inputs
/// * `folder_img_path`: Path to the directory containing images to load as tensors.
/// * `model`: A neural network model implementing the `Module` trait.
/// * `(resize_h, resize_w)`: Target height and width for image resizing.
/// * `batch_size`: Number of samples processed per batch.
/// * `f_class_list`: Path to a file containing class names (one per line).
/// * `(storage_device, compute_device)`: Tuple specifying storage (CPU/GPU) and compute devices.
///
/// # Output
/// A `PredictionsDataframe` struct containing:
/// - Image names
/// - Class names
/// - Prediction tensors
pub fn calc_inferences_for_dataset(
    folder_img_path: &Path,
    model: &dyn Module,
    (resize_h, resize_w): (usize, usize),
    batch_size: usize,
    f_class_list: &str,
    (storage_device, compute_device): (Device, Device),
) -> cResult<PredictionsDataframe> {
    println!("Appel à calc_inferences_for_dataset");

    let transform518 = ImgToTensorResizeImageNetNormTransform {
        resize_h: Some(resize_h),
        resize_w: Some(resize_w),
        device: storage_device.clone(),
    };

    let dataset = SimpleDataset::new(folder_img_path, Box::new(transform518));
    let n_images = dataset.len();
    println!("Dataset chargé");

    let dataloader = InferenceDataloader::new(Box::new(dataset), batch_size);
    let n_batchs = dataloader.len();
    println!("Dataloader chargé");

    let class_list = read_text_file(f_class_list);

    println!("Lancement des inférences");
    let verbose = true;
    let (ids, tensor) = calc_inferences_from_iterator(
        dataloader,
        model,
        n_batchs,
        n_images,
        verbose,
        &storage_device,
        &compute_device,
    )?;

    println!("Inférences terminées");

    Ok(PredictionsDataframe {
        class_list,
        lines: ids
            .into_iter()
            .map(|s| get_basename(&s).expect("Error"))
            .zip(TensorIterator::new(tensor))
            .collect(),
    })
}

/// Tiling inference: Computes deep features for a dataset and saves them to disk.
pub fn calc_inferences_tiling_deep_features(
    d_in_dataset_folder: &Path,
    d_out_features_folder: &Path,
    b_use_pc24model: bool,
    tile_batch_size: usize,
    res_base: usize,    // Exemple 3 * 518;
    geo: Vec<Geometry>, // = vec![(0, 0, 0, 0)];
    b_load_img_with_wand: bool,
) -> cResult<()> {
    println!("Entrée dans calc_inferences_tiling_deep_features..");
    println!("  d_in_dataset_folder: {d_in_dataset_folder:?}");
    println!("  d_out_features_folder: {d_out_features_folder:?}");
    println!("  tile_batch_size: {tile_batch_size}");

    let ext_out = ".safetensors";

    let use_cpu = false;
    let storage_device = candle_core::Device::Cpu;
    let compute_device = get_compute_device(use_cpu).expect("Error");
    println!("Devices chargés");

    // Load the DINO model, wrapped in ForwardFeaturesModel to compute only deep features
    //let modele_complet: Box<dyn FeaturesModule> = Box::new(load_model_dinov2(b_use_pc24model, &compute_device)?);
    //let features_model = FeaturesModel { features_module : modele_complet.as_ref()};
    let modele_complet: &dyn FeaturesModule = &load_model_dinov2(b_use_pc24model, &compute_device)?;
    let features_model = FeaturesModel {
        features_module: modele_complet,
    };
    println!("Modèle chargé");

    if let Err(e) = fs::create_dir_all(d_out_features_folder) {
        panic!(
            "Erreur lors de la création du dossier {:?} : {:?}",
            d_out_features_folder, e
        );
    }

    let n_plots = get_folder_images_list(d_in_dataset_folder)
        .expect("Error in get_folder_images_list")
        .count();

    println!("Début de la boucle principale sur les plots-images");
    let t_start = seconds_since_epoch();

    let init_n_plots_done = get_folder_tensors_list(d_out_features_folder)
        .expect("Error in get_folder_tensors_list")
        .count();

    // Main loop: iterates over plot-images in the dataset (lazy loading, no Dataloader)
    // Skips image loading if the output tensor already exists
    // (Computations can be stopped and resumed later)
    let mut n_skip: Option<usize> = Some(0);
    for (plot_index, p_in) in get_folder_images_list(d_in_dataset_folder)
        .expect("Error in get_folder_images_list")
        .enumerate()
    {
        // Build output filename
        let p_out = d_out_features_folder.join(
            get_basename(&p_in.to_string_lossy().into_owned())
                .expect("Error basename")
                .replace(".jpg", ext_out)
                .replace(".bmp", ext_out),
        );

        if p_out.exists() {
            match n_skip {
                None => {
                    println!("Warning: Images are not loaded in the same order than before");
                }
                Some(n) => n_skip = Some(n + 1),
            }
            continue;
        } else if let Some(n) = n_skip {
            if n > 0 {
                println!("Previous plot(s) already computed: {n}");
            }
            n_skip = None;
        }

        println!("plot {plot_index}/{n_plots} : {p_in:?}");

        // Create the dataset to serve plot tiles and compute their geometry for tiling
        let img_path = p_in
            .as_path()
            .as_os_str()
            .to_str()
            .expect("Le chemin contient des caractères non valides en UTF-8");
        let tiles_ds: Box<dyn Dataset<(usize, Tensor)>> = if b_load_img_with_wand {
            Box::new(
                WandTilesDataset::init_with_geometry(img_path, res_base, &geo, &storage_device)
                    .expect("Error creating Tiles dataset"),
            )
        } else {
            Box::new(
                TilesDataset::init_with_geometry(img_path, res_base, &geo, &storage_device)
                    .expect("Error creating Tiles dataset"),
            )
        };
        let n_tiles = tiles_ds.len();

        // Create the dataloader to serve tiles for batch processing
        let tiles_loader = InferenceDataloader::new(tiles_ds, tile_batch_size);
        let n_batch = tiles_loader.len();

        println!("Plot Tiling -> {n_tiles} tiles et {n_batch} batchs (bs {tile_batch_size}).");

        // Compute inferences to extract deep features from the input tiles
        let verbose = false;
        let (_ids, tensor) = calc_inferences_from_iterator(
            tiles_loader,
            &features_model,
            n_batch,
            n_tiles,
            verbose,
            &storage_device,
            &compute_device,
        )?;

        // Post-inference: discard IDs, retain only the output tensor, and save it to disk with the appropriate filename
        save_tensor(&tensor, Path::new(&p_out));

        let t_after_inference = seconds_since_epoch();
        let cour_n_plots_done = get_folder_tensors_list(d_out_features_folder)
            .expect("Error in get_folder_tensors_list")
            .count();
        let duration = t_after_inference - t_start;

        let nb_seconds_per_plot = duration / ((cour_n_plots_done - init_n_plots_done) as f64);
        let nb_plots_per_min = 60. / nb_seconds_per_plot;
        let remaining_time_m = (n_plots - cour_n_plots_done) as f64 / nb_plots_per_min;
        let remaining_time_h = remaining_time_m / 60.;
        let remaining_time_d = remaining_time_h / 24.;

        println!();
        println!("**Perfs**: {nb_plots_per_min:.2} plots/min ; {nb_seconds_per_plot:.2}s/plot");
        print!("Temps restant estimé: ");
        if (1.0..).contains(&remaining_time_d) {
            print!("{remaining_time_d:.2}jours , ");
        }
        if (1.0..100.).contains(&remaining_time_h) {
            print!("{remaining_time_h:.2}h , ");
        }
        if (1.0..100.).contains(&remaining_time_m) {
            print!("{remaining_time_m:.2}min , ");
        }
        println!();
        println!();
    }

    println!("Calcul des deep features terminé pour le tiling");

    Ok(())
}

/// Computes classification head predictions from precomputed deep features stored in `d_in_features_folder`.
pub fn calc_inferences_tiling_head(
    d_in_features_folder: &Path,
    b_use_pc24model: bool,
    f_class_list: &str,
) -> cResult<PredictionsDataframe> {
    println!("Appel à calc_inferences_tiling_head");

    let use_cpu = false;
    let storage_device = candle_core::Device::Cpu;
    let compute_device = get_compute_device(use_cpu).expect("Error");
    println!("Devices chargés");

    let dataset = TensorDataset::new(d_in_features_folder, storage_device.clone());
    let n_images = dataset.len();
    println!("Dataset chargé");

    // Variable tile count per image (due to differing plot sizes)
    // results in non-concatenatable tensors => hence batch_size=1
    let batch_size = 1;
    let dataloader = InferenceDataloader::new(Box::new(dataset), batch_size);
    let n_batchs = dataloader.len();
    println!("Dataloader chargé");

    // Load the DINO model, wrapped in TilingHeadModel to aggregate tile-level predictions at the plot scale
    let model_complet = load_model_dinov2(b_use_pc24model, &compute_device)?;
    let model = TilingHeadModel {
        head_model: &model_complet,
    };
    println!("Modèle chargé");

    let class_list = read_text_file(f_class_list);

    println!("Lancement des inférences");
    let verbose = true;
    let (ids, tensor) = calc_inferences_from_iterator(
        dataloader,
        &model,
        n_batchs,
        n_images,
        verbose,
        &storage_device,
        &compute_device,
    )?;

    println!("Inférences terminées");

    Ok(PredictionsDataframe {
        class_list,
        lines: ids
            .into_iter()
            .map(|s| get_basename(&s).expect("Error"))
            .zip(TensorIterator::new(tensor))
            .collect(),
    })
}

/// VaMIS inference: Computes deep features (possibly with window shifted attention) for high-resolution images.
/// Output tensors are saved to `d_out_features_folder` for lazy resuming.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::let_and_return)]
pub fn calc_inferences_vamis_deep_features(
    d_in_dataset_folder: &Path,
    d_out_features_folder: &Path,
    b_use_pc24model: bool,
    b_use_pc24wsa: bool,
    variable_model_input_size: usize, // Taille VaMIS: Exemple 1554
    window_size_pixels: usize, // Must be a multiple of 14 (e.g., 518 = 37*14) and a divisor of the VaMIS input size
    global_attn_indexes: &[usize],
    attn_cuda_type: usize,
    b_use_wand_for_image_loading: bool,
) -> cResult<()> {
    let patch_size = 14;

    println!("Call to calc_inferences_vamis_deep_features..");
    println!("  d_in_dataset_folder      : {d_in_dataset_folder:?}");
    println!("  d_out_features_folder    : {d_out_features_folder:?}");
    println!("  b_use_pc24model          : {b_use_pc24model}");
    println!("  b_use_pc24wsa            : {b_use_pc24wsa}");
    println!("  Variable model_input_size: {variable_model_input_size}");
    println!("  window_size_pixels       : {window_size_pixels}");
    println!("  global_attn_indexes      : {global_attn_indexes:?}");
    println!("  attn_cuda_type           : {attn_cuda_type}");
    // println!("  batch_size               : {batch_size}");

    let ext_out = ".safetensors";

    let use_cpu = false;
    //let storage_device = candle_core::Device::Cpu;
    let compute_device = get_compute_device(use_cpu).expect("Error");
    println!("Devices chargés");

    // Load the DINO model, wrapped in ForwardFeaturesModel to extract only deep features (no classification head)
    assert_eq!(window_size_pixels % patch_size, 0); // 518 % 14   = 0
    assert_eq!(variable_model_input_size % window_size_pixels, 0); // 1554 % 518 = 0
    let window_size = window_size_pixels / patch_size;

    let modele_complet: &dyn FeaturesModule = if b_use_pc24wsa {
        &load_model_dinov2_base_pc24_wsa(
            &compute_device,
            window_size,
            global_attn_indexes,
            attn_cuda_type,
        )
        .expect("Error loading load_model_dinov2_base_pc24_wsa")
    } else {
        let b_use_pc24model = true;
        &(load_model_dinov2(b_use_pc24model, &compute_device)
            .expect("Error loading load_model_dinov2"))
    };

    let model = FeaturesModel {
        features_module: modele_complet,
    };
    println!("Modèle chargé");

    if let Err(e) = fs::create_dir_all(d_out_features_folder) {
        panic!("Erreur lors de la création du dossier : {:?}", e);
    }

    let n_plots = get_folder_images_list(d_in_dataset_folder)
        .expect("Error in get_folder_images_list")
        .count();

    println!("Début de la boucle principale sur les plots-images");
    let t_start = seconds_since_epoch();

    let init_n_plots_done = get_folder_tensors_list(d_out_features_folder)
        .expect("Error in get_folder_tensors_list")
        .count();

    // Main loop: iterate over plot images in the dataset (lazy loading, no Dataloader)
    // Skip image processing if the output tensor already exists (enables interruptible workflows)
    let mut n_skip: Option<usize> = Some(0);
    for (plot_index, p_in) in get_folder_images_list(d_in_dataset_folder)
        .expect("Error in get_folder_images_list")
        .enumerate()
    {
        // Build output filename
        let p_out = d_out_features_folder.join(
            get_basename(&p_in.to_string_lossy().into_owned())
                .expect("Error basename")
                .replace(".jpg", ext_out)
                .replace(".bmp", ext_out),
        );

        if p_out.exists() {
            match n_skip {
                None => {
                    println!("Warning: Images are not loaded in the same order than before");
                }
                Some(n) => n_skip = Some(n + 1),
            }
            continue;
        } else if let Some(n) = n_skip {
            if n > 0 {
                println!("Previous plot(s) already computed: {}", n);
            }
            n_skip = None;
        }

        let plot_index_inc = plot_index + 1;
        println!("plot {plot_index_inc}/{n_plots} : {p_in:?}");

        // Initialize the dataset to serve plot tiles and compute their spatial geometry for tiling
        let img_path = p_in
            .as_path()
            .as_os_str()
            .to_str()
            .expect("Le chemin contient des caractères non valides en UTF-8");

        // Read image, perform inference
        let tensor_img = if b_use_wand_for_image_loading {
            let img = load_image_and_resize_with_wand3(img_path, variable_model_input_size)
                .expect("Error loading image with wand");
            let transform = ImgToTensorResizeImageNetNormTransform {
                resize_h: None,
                resize_w: None,
                device: compute_device.clone(),
            };

            let tensor_img = transform.apply(&img).expect("Error with transform_apply");
            tensor_img
        } else {
            let img = load_image(&p_in).expect("Error loading image with wand");
            let transform = ImgToTensorResizeImageNetNormTransform {
                resize_h: Some(variable_model_input_size),
                resize_w: Some(variable_model_input_size),
                device: compute_device.clone(),
            };

            let tensor_img = transform.apply(&img).expect("Error with transform_apply");
            tensor_img
        };

        let tensor = model.forward(&tensor_img).expect("Error in model forward");

        // Post-inference: discard sample IDs, retain only the output tensor, and persist it to disk at the designated path
        save_tensor(&tensor, Path::new(&p_out));

        let t_after_inference = seconds_since_epoch();
        let cour_n_plots_done = get_folder_tensors_list(d_out_features_folder)
            .expect("Error in get_folder_tensors_list")
            .count();
        let duration = t_after_inference - t_start;

        let nb_seconds_per_plot = duration / ((cour_n_plots_done - init_n_plots_done) as f64);
        let nb_plots_per_min = 60. / nb_seconds_per_plot;
        let remaining_time_m = (n_plots - cour_n_plots_done) as f64 / nb_plots_per_min;
        let remaining_time_h = remaining_time_m / 60.;
        let remaining_time_d = remaining_time_h / 24.;

        println!();
        println!("**Perfs**: {nb_plots_per_min:.2} plots/min ; {nb_seconds_per_plot:.2}s/plot");
        print!("Temps restant estimé: ");
        if (1.0..).contains(&remaining_time_d) {
            print!("{remaining_time_d:.2}jours , ");
        }
        if (1.0..100.).contains(&remaining_time_h) {
            print!("{remaining_time_h:.2}h , ");
        }
        if (1.0..100.).contains(&remaining_time_m) {
            print!("{remaining_time_m:.2}min , ");
        }
        println!();
        println!();
    }

    println!("Calcul des deep features terminé pour vamis");

    Ok(())
}
