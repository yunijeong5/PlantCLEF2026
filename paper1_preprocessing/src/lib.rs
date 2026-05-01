use candle_core::{D, Device, IndexOp, Module, Result as cResult, Tensor};
use std::collections::HashSet;
use std::path::Path;

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};

pub mod dinov2reg4_mod;
pub mod dinov2reg4_wsa;
pub mod lib_aggregation;
pub mod lib_dataset;
pub mod lib_files;
pub mod lib_inference;
pub mod lib_model;
pub mod lib_soumission;

use lib_dataset::{Geometry, ImgToTensorResizeImageNetNormTransform, Transform};
use lib_files::{PredictionsDataframe, load_image_and_resize_with_wand3};

pub const D_DATASTORE: &str = "data";

/// Infers a single image using a DINOv2 model.
///
/// # Arguments
/// * `f_img_in` - Path to the input image.
/// * `use_cpu` - Use CPU for computations.
/// * `b_use_pc24model` - Use the DINOv2Reg4 PC24 model.
/// * `model_input_size` - Input size of the model.
/// * `b_use_pc24wsa` - Use the model with Window Shifted Attention.
/// * `window_size` - Window size for WSA.
/// * `global_attn_indexes` - Indexes of layers with global attention.
/// * `attn_cuda_type` - Type of CUDA attention.
/// * `f_species_id_mapping` - Path to the species ID mapping file.
///
/// # Returns
/// A tuple containing the names of the top 5 most probable species and their probabilities.
#[allow(clippy::too_many_arguments)]
pub fn inference_single_image(
    f_img_in: &str,
    use_cpu: bool,
    b_use_pc24model: bool,
    model_input_size: usize,
    b_use_pc24wsa: bool,
    window_size: usize,
    global_attn_indexes: &[usize],
    attn_cuda_type: usize,
    f_species_id_mapping: &str,
) -> cResult<(Vec<String>, Vec<f32>)> {
    let device = lib_inference::get_compute_device(use_cpu).expect("Error");
    let img = load_image_and_resize_with_wand3(f_img_in, model_input_size)
        .expect("Image file not found.");
    let transform = ImgToTensorResizeImageNetNormTransform {
        resize_h: Some(model_input_size),
        resize_w: Some(model_input_size),
        device: device.clone(),
    };
    let img_tensor = transform.apply(&img).expect("Error applying transform");

    let classes: Vec<String> = std::fs::read_to_string(f_species_id_mapping)
        .expect("Missing classes file")
        .split('\n')
        .map(|s| s.to_string())
        .collect();

    let model: &dyn Module = if b_use_pc24wsa {
        &lib_model::load_model_dinov2_base_pc24_wsa(
            &device,
            window_size,
            global_attn_indexes,
            attn_cuda_type,
        )
        .expect("Error loading load_model_dinov2_base_pc24_wsa")
    } else {
        &(lib_model::load_model_dinov2(b_use_pc24model, &device)
            .expect("Error loading load_model_dinov2"))
    };

    let logits = model.forward(&img_tensor)?;
    let prs = candle_nn::ops::softmax(&logits, D::Minus1)?
        .i(0)?
        .to_vec1::<f32>()?;
    let mut prs = prs.iter().enumerate().collect::<Vec<_>>();
    prs.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));

    // Retrieve the top 5 most probable species
    let top_five: Vec<_> = prs.into_iter().take(5).collect();
    let top_species: Vec<String> = top_five
        .iter()
        .map(|(category_idx, _)| classes[*category_idx].clone())
        .collect();
    let top_probs: Vec<f32> = top_five.iter().map(|(_, pr)| **pr).collect();

    if false {
        // Display results
        println!("Top 5 species and probabilities:");
        for (species, prob) in top_species.iter().zip(top_probs.iter()) {
            println!("{:24}: {:.2}%", species, 100. * prob);
        }
    }

    Ok((top_species, top_probs))
}

/// Baseline518: Computes inferences and predictions for a given dataset without tiling.
///
/// # Arguments
/// * `tag_dataset` - Tag of the dataset.
/// * `d_sub_dataset_folder` - Sub-dataset folder.
/// * `tag_sup` - Additional tag for output files.
/// * `tag_date` - Date for output files.
/// * `batch_size` - Batch size.
/// * `model_input_size` - Input size of the model.
/// * `top_k` - Number of classes to keep.
/// * `min_score` - Minimum score for predictions.
/// * `b_write_csv_submission` - Write the submission file.
/// * `use_cpu_for_computations` - Use CPU for computations.
///
/// # Returns
/// Result of the operation.
#[allow(clippy::too_many_arguments)]
pub fn calc_inferences_518_predictions_et_soumission(
    tag_dataset: &str,
    d_sub_dataset_folder: &str,
    tag_sup: &str,
    tag_date: &str,
    batch_size: usize,
    model_input_size: usize,
    top_k: usize,
    min_score: f32,
    b_write_csv_submission: bool,
    use_cpu_for_computations: bool,
) -> cResult<()> {
    let f_out_csv_predictions = format!(
        "{}/50_probas_predictions_csv/{}_{}_predictions_{}.csv",
        D_DATASTORE, tag_dataset, tag_date, tag_sup
    );
    let f_out_csv_submission = format!(
        "{}/65_submissions/{}_{}_submission_{}.csv",
        D_DATASTORE, tag_dataset, tag_date, tag_sup
    );
    let d_in_dataset_folder = format!(
        "{}/10_images/{}/{}",
        D_DATASTORE, d_sub_dataset_folder, tag_dataset
    );
    let p_in_dataset_folder = Path::new(&d_in_dataset_folder);
    let f_in_csv_species_ids = format!("{}/30_models/species_id_mapping2.txt", D_DATASTORE);

    let compute_device =
        lib_inference::get_compute_device(use_cpu_for_computations).expect("Error");
    let storage_device = Device::Cpu;

    let b_use_pc24model = true;
    let model = lib_model::load_model_dinov2(b_use_pc24model, &compute_device)
        .expect("Error loading model");

    let df_predictions = lib_inference::calc_inferences_for_dataset(
        p_in_dataset_folder,
        &model,
        (model_input_size, model_input_size),
        batch_size,
        &f_in_csv_species_ids,
        (storage_device.clone(), compute_device),
    )
    .expect("Error");

    df_predictions
        .to_csv(&f_out_csv_predictions)
        .expect("Error creating CSV");

    if b_write_csv_submission {
        let b_apply_softmax = true;
        lib_soumission::calc_submission_file_from_predictions_file(
            &f_out_csv_predictions,
            &f_out_csv_submission,
            top_k,
            min_score,
            b_apply_softmax,
            &storage_device,
        )
        .expect("Error in calc_submission_file_from_predictions_file");
    }

    Ok(())
}

/// Computes deep features for tiling at a given scale.
///
/// # Arguments
/// * `tag_dataset` - Tag of the dataset.
/// * `d_out_features_folder` - Output folder for features.
/// * `scale` - Scale of the tiling.
/// * `b_use_pc24model` - Use the DINOv2Reg4 PC24 model.
/// * `tile_batch_size` - Batch size for tiles.
/// * `res_base` - Base resolution.
/// * `b_load_img_with_wand` - Load images with Wand.
///
/// # Returns
/// Result of the operation.
pub fn calc_deep_features_for_tiling_for_given_scale(
    tag_dataset: &str,
    d_out_features_folder: &Path,
    scale: usize,
    b_use_pc24model: bool,
    tile_batch_size: usize,
    res_base: usize,
    b_load_img_with_wand: bool,
) -> cResult<()> {
    let s_in_dataset_folder = format!("{}/10_images/PlantCLEF2025/{}/", D_DATASTORE, tag_dataset);
    let d_in_dataset_folder = Path::new(&s_in_dataset_folder);
    let model_input_size = 518;

    let mut geo: Vec<Geometry> = Vec::new();
    for i in 0..scale {
        for j in 0..scale {
            geo.push(Geometry {
                height: model_input_size,
                width: model_input_size,
                height_offset: (model_input_size * i).try_into().unwrap(),
                width_offset: (model_input_size * j).try_into().unwrap(),
            });
        }
    }

    lib_inference::calc_inferences_tiling_deep_features(
        d_in_dataset_folder,
        d_out_features_folder,
        b_use_pc24model,
        tile_batch_size,
        res_base,
        geo,
        b_load_img_with_wand,
    )
    .expect("Error in calc_inferences_tiling_deep_features");

    Ok(())
}

/// Computes deep features for tiling at multiple scales.
///
/// # Arguments
/// * `tag_dataset` - Tag of the dataset.
/// * `scale_list` - List of scales.
/// * `b_load_img_with_wand` - Load images with Wand.
///
/// # Returns
/// Result of the operation.
pub fn calc_deep_features_for_tiling(
    tag_dataset: &str,
    scale_list: Vec<usize>,
    b_load_img_with_wand: bool,
) -> cResult<()> {
    for scale in scale_list {
        let tag_df = if b_load_img_with_wand {
            format!("{}_tiling{}x{}_backendwand/", tag_dataset, &scale, &scale)
        } else {
            format!("{}_tiling{}x{}_backendimage/", tag_dataset, &scale, &scale)
        };
        let s_out_features_folder = format!("{}/20_deep_features/rust/{}", D_DATASTORE, tag_df);
        let d_out_features_folder = Path::new(&s_out_features_folder);
        calc_deep_features_for_tiling_for_given_scale(
            tag_dataset,
            d_out_features_folder,
            scale,
            true,
            128,
            518 * scale,
            b_load_img_with_wand,
        )?;
    }

    Ok(())
}

/// Computes predictions from deep features for a given scale.
///
/// # Arguments
/// * `scale` - Scale.
/// * `tag_dataset` - Tag of the dataset.
/// * `tag_date` - Date for output files.
/// * `b_load_img_with_wand` - Load images with Wand.
/// * `b_write_csv_submission` - Write the submission file.
/// * `top_k` - Number of classes to keep.
/// * `min_score` - Minimum score for predictions.
///
/// # Returns
/// Result of the operation.
pub fn calc_tiling_predictions_from_deep_features_for_given_scale(
    scale: usize,
    tag_dataset: &str,
    tag_date: &str,
    b_load_img_with_wand: bool,
    b_write_csv_submission: bool,
    top_k: usize,
    min_score: f32,
) -> cResult<()> {
    let tag_df = if b_load_img_with_wand {
        format!("{}_tiling{}x{}_backendwand", tag_dataset, &scale, &scale)
    } else {
        format!("{}_tiling{}x{}_backendimage", tag_dataset, &scale, &scale)
    };

    let d_in_features_folder = format!("{}/20_deep_features/rust/{}", D_DATASTORE, tag_df);

    let f_out_csv_predictions = format!(
        "{}/50_probas_predictions_csv/{}_{}_rust_predictions.csv",
        D_DATASTORE, tag_df, tag_date
    );
    let f_out_csv_submission = format!(
        "{}/50_probas_predictions_csv/{}_{}_rust_submission_threshold_{:02.1}prcts.csv",
        D_DATASTORE,
        tag_df,
        tag_date,
        min_score * 100.
    );

    if Path::new(&f_out_csv_predictions).metadata().is_err() {
        let f_class_list = format!("{}/30_models/species_id_mapping2.txt", D_DATASTORE);

        let df_predictions = lib_inference::calc_inferences_tiling_head(
            Path::new(&d_in_features_folder),
            true,
            &f_class_list,
        )
        .expect("Error in calc_inferences_tiling_head");

        df_predictions
            .to_csv(&f_out_csv_predictions)
            .expect("Error in exporting predictions");
    }

    if b_write_csv_submission {
        let b_apply_softmax = false;
        let storage_device = Device::Cpu;
        lib_soumission::calc_submission_file_from_predictions_file(
            &f_out_csv_predictions,
            &f_out_csv_submission,
            top_k,
            min_score,
            b_apply_softmax,
            &storage_device,
        )
        .expect("Error in calc_submission_file_from_predictions_file");
    }

    Ok(())
}

/// Computes predictions from deep features for multiple scales.
///
/// # Arguments
/// * `tag_dataset` - Tag of the dataset.
/// * `tag_date` - Date for output files.
/// * `scale_list` - List of scales.
/// * `b_load_img_with_wand` - Load images with Wand.
///
/// # Returns
/// Result of the operation.
pub fn calc_tiling_predictions_from_deep_features(
    tag_dataset: &str,
    tag_date: &str,
    scale_list: Vec<usize>,
    b_load_img_with_wand: bool,
) -> cResult<()> {
    for scale in scale_list {
        calc_tiling_predictions_from_deep_features_for_given_scale(
            scale,
            tag_dataset,
            tag_date,
            b_load_img_with_wand,
            false,
            15,
            0.01,
        )?;
    }

    Ok(())
}

/// Concatenates prediction CSV files for each scale into a single file.
/// Does not parse the CSVs, just appends the lines (without headers after the first file).
///
/// # Arguments
/// * `tag_dataset` - Tag of the dataset.
/// * `scale_list` - List of scales.
/// * `b_load_img_with_wand` - Load images with Wand.
/// * `tag_date` - Date for output files.
///
/// # Returns
/// Path to the concatenated CSV file.
pub fn concat_predictions_csv_for_scales(
    tag_dataset: &str,
    scale_list: Vec<usize>,
    b_load_img_with_wand: bool,
    tag_date: &str,
) -> cResult<String> {
    let n_tiles: usize = scale_list.iter().map(|&x| x * x).sum();
    let tag_df = if b_load_img_with_wand {
        format!("{}_tiling{}_backendwand", tag_dataset, &n_tiles)
    } else {
        format!("{}_tiling{}_backendimage", tag_dataset, &n_tiles)
    };
    let output_file = format!(
        "{}/50_probas_predictions_csv/{}_{}_rust_predictions_concat.csv",
        D_DATASTORE, tag_df, tag_date
    );

    let mut first_file = true;
    for scale in scale_list {
        let tag_df = if b_load_img_with_wand {
            format!("{}_tiling{}x{}_backendwand", tag_dataset, scale, scale)
        } else {
            format!("{}_tiling{}x{}_backendimage", tag_dataset, scale, scale)
        };
        let input_file = format!(
            "{}/50_probas_predictions_csv/{}_{}_rust_predictions.csv",
            D_DATASTORE, tag_df, tag_date
        );

        let mut output = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&output_file)?;

        let input = File::open(&input_file)?;
        let reader = BufReader::new(input);

        for (i, line) in reader.lines().enumerate() {
            let line = line?;
            // Skip header for files after the first
            if first_file || i > 0 {
                writeln!(output, "{}", line)?;
            }
        }
        first_file = false;
    }

    Ok(output_file)
}

/// Aggregates predictions by tile to obtain predictions by plot.
///
/// # Arguments
/// * `scale_list` - List of scales.
/// * `tag_dataset` - Tag of the dataset.
/// * `tag_date` - Date for output files.
/// * `min_score` - Minimum score for predictions.
/// * `top_k` - Number of classes to keep.
/// * `b_apply_softmax_in_aggregation` - Apply SoftMax during aggregation.
/// * `b_apply_softmax_for_submission` - Apply SoftMax for submission.
/// * `b_write_csv_submission` - Write the submission file.
/// * `use_cpu_for_computations` - Use CPU for computations.
///
/// # Returns
/// Result of the operation.
#[allow(clippy::too_many_arguments)]
pub fn aggrege_tiling_manuel_predictions_par_tuile_et_calcule_predictions_par_plot_et_soumission(
    scale_list: Vec<usize>,
    tag_dataset: &str,
    tag_date: &str,
    min_score: f32,
    top_k: usize,
    b_load_img_with_wand: bool,
    b_apply_softmax_in_aggregation: bool,
    b_apply_softmax_for_submission: bool,
    b_write_csv_submission: bool,
    _use_cpu_for_computations: bool,
) -> cResult<()> {
    let n_tiles: usize = scale_list.iter().map(|&x| x * x).sum();
    let tag_df = if b_load_img_with_wand {
        format!("{}_tiling{}_backendwand", tag_dataset, &n_tiles)
    } else {
        format!("{}_tiling{}_backendimage", tag_dataset, &n_tiles)
    };

    let f_in_csv_predictions_par_tuile = format!(
        "{}/50_probas_predictions_csv/{}_{}_rust_predictions_concat.csv",
        D_DATASTORE, tag_df, tag_date
    );
    let f_out_csv_predictions = format!(
        "{}/50_probas_predictions_csv/{}_{}_rust_predictions.csv",
        D_DATASTORE, tag_df, tag_date
    );
    let f_out_csv_submission = format!(
        "{}/65_submissions/{}_{}_submission_seuil_{:02.1}prcts.csv",
        D_DATASTORE,
        tag_df,
        tag_date,
        min_score * 100.
    );

    let storage_device = Device::Cpu;

    if Path::new(&f_out_csv_predictions).metadata().is_err() {
        let df_in_predictions_par_tuile =
            PredictionsDataframe::from_csv(&f_in_csv_predictions_par_tuile, &storage_device)
                .expect("Error");

        let class_list: Vec<String> = df_in_predictions_par_tuile.class_list;

        let ids: Vec<String> = df_in_predictions_par_tuile
            .lines
            .iter()
            .map(|(s, _t)| String::from(s.split('_').next().expect("Missing underscore in name _")))
            .collect();

        let unique_ids: HashSet<String> = ids.into_iter().collect();
        let mut ids: Vec<String> = unique_ids.into_iter().collect();
        ids.sort();

        let lines: Vec<(String, Tensor)> = ids
            .iter()
            .map(|id| {
                let filtered_tensors: Vec<Tensor> = df_in_predictions_par_tuile
                    .lines
                    .iter()
                    .filter(|(s, _)| s.contains(id))
                    .map(|(_, t)| t.unsqueeze(0).expect("Error"))
                    .collect();
                let concatenated_tensor =
                    Tensor::cat(&filtered_tensors, 0).expect("Error in concatenation");

                let probas_plot = if b_apply_softmax_in_aggregation {
                    candle_nn::ops::softmax(&concatenated_tensor, D::Minus1).expect("Error")
                } else {
                    concatenated_tensor
                };

                let probas_plot = probas_plot.max(0).expect("Error calculating max");
                (String::from(id), probas_plot)
            })
            .collect();

        let df_out_predictions_par_plot = PredictionsDataframe { class_list, lines };
        df_out_predictions_par_plot
            .to_csv(&f_out_csv_predictions)
            .expect("Error creating CSV");
    }

    if b_write_csv_submission {
        lib_soumission::calc_submission_file_from_predictions_file(
            &f_out_csv_predictions,
            &f_out_csv_submission,
            top_k,
            min_score,
            b_apply_softmax_for_submission,
            &storage_device,
        )
        .expect("Error in calc_submission_file_from_predictions_file");
    }

    Ok(())
}

pub fn main_calc_tiling(
    tag_dataset: &str,
    tag_date: &str,
    scale_list: Vec<usize>,
    b_load_img_with_wand: bool,
    top_k: usize,
    min_score: f32,
) -> cResult<()> {
    // 1. Compute deep features for all scales
    calc_deep_features_for_tiling(tag_dataset, scale_list.clone(), b_load_img_with_wand)?;

    // 2. Compute predictions for each scale
    calc_tiling_predictions_from_deep_features(
        tag_dataset,
        tag_date,
        scale_list.clone(),
        b_load_img_with_wand,
    )?;

    // 3. Concatenate prediction CSVs
    let _f_concat_csv = concat_predictions_csv_for_scales(
        tag_dataset,
        scale_list.clone(),
        b_load_img_with_wand,
        tag_date,
    )?;

    // 4. Aggregate predictions (max-pooling by plot)
    aggrege_tiling_manuel_predictions_par_tuile_et_calcule_predictions_par_plot_et_soumission(
        scale_list.clone(),
        tag_dataset,
        tag_date,
        min_score,
        top_k,
        b_load_img_with_wand,
        false, // b_apply_softmax_in_aggregation
        false, // b_apply_softmax_for_submission
        true,  // b_write_csv_submission
        true,  // use_cpu_for_computations
    )?;

    Ok(())
}

/// Computes inferences, predictions, and submission with VaMIS model and WindowShiftedAttention.
///
/// # Arguments
/// * `model_input_size` - Input size of the model.
/// * `window_size` - Window size for WSA.
/// * `global_attn_indexes` - Indexes of layers with global attention.
/// * `b_use_pc24wsa` - Use the model with Window Shifted Attention.
/// * `attn_cuda_type` - Type of CUDA attention.
/// * `tag_dataset` - Tag of the dataset.
/// * `d_sub_dataset_folder` - Sub-dataset folder.
/// * `tag_sup` - Additional tag for output files.
/// * `tag_date` - Date for output files.
/// * `batch_size` - Batch size.
/// * `top_k` - Number of classes to keep.
/// * `min_score` - Minimum score for predictions.
/// * `use_cpu_for_computations` - Use CPU for computations.
///
/// # Returns
/// Result of the operation.
#[allow(clippy::too_many_arguments)]
pub fn calc_vamis_wsa_inferences_predictions_et_soumission(
    model_input_size: usize,
    window_size: usize,
    global_attn_indexes: &[usize],
    b_use_pc24wsa: bool,
    attn_cuda_type: usize,
    tag_dataset: &str,
    d_sub_dataset_folder: &str,
    tag_sup: &str,
    tag_date: &str,
    batch_size: usize,
    top_k: usize,
    min_score: f32,
    use_cpu_for_computations: bool,
) -> cResult<()> {
    let f_out_csv_predictions = format!(
        "{}/50_probas_predictions_csv/{}_{}_predictions_{}.csv",
        D_DATASTORE, tag_dataset, tag_date, tag_sup
    );
    let f_out_csv_submission = format!(
        "{}/65_submissions/{}_{}_submission_{}_seuil_{:02.1}prcts.csv",
        D_DATASTORE,
        tag_dataset,
        tag_date,
        tag_sup,
        min_score * 100.
    );

    let d_in_dataset_folder = format!(
        "{}/10_images/{}/{}",
        D_DATASTORE, d_sub_dataset_folder, tag_dataset
    );
    let p_in_dataset_folder = Path::new(&d_in_dataset_folder);
    let f_in_csv_species_ids = format!("{}/30_models/species_id_mapping2.txt", D_DATASTORE);

    let compute_device =
        lib_inference::get_compute_device(use_cpu_for_computations).expect("Error");
    let storage_device = Device::Cpu;

    if Path::new(&f_out_csv_predictions).metadata().is_err() {
        let model: &dyn Module = if b_use_pc24wsa {
            &lib_model::load_model_dinov2_base_pc24_wsa(
                &compute_device,
                window_size,
                global_attn_indexes,
                attn_cuda_type,
            )
            .expect("Error loading load_model_dinov2_base_pc24_wsa")
        } else {
            &(lib_model::load_model_dinov2(true, &compute_device)
                .expect("Error loading load_model_dinov2"))
        };

        let df_predictions = lib_inference::calc_inferences_for_dataset(
            p_in_dataset_folder,
            model,
            (model_input_size, model_input_size),
            batch_size,
            &f_in_csv_species_ids,
            (storage_device.clone(), compute_device),
        )
        .expect("Error");

        df_predictions
            .to_csv(&f_out_csv_predictions)
            .expect("Error creating CSV");
    }

    if Path::new(&f_out_csv_predictions).metadata().is_ok() {
        let b_apply_softmax = true;
        lib_soumission::calc_submission_file_from_predictions_file(
            &f_out_csv_predictions,
            &f_out_csv_submission,
            top_k,
            min_score,
            b_apply_softmax,
            &storage_device,
        )
        .expect("Error in calc_submission_file_from_predictions_file");
    }

    Ok(())
}

/// Computes deep features for VaMIS.
///
/// # Arguments
/// * `b_use_wand_for_image_loading` - Load images with Wand.
/// * `variable_model_input_size` - Variable input size of the model.
/// * `window_size_pixels` - Window size in pixels.
/// * `global_attn_indexes` - Indexes of layers with global attention.
/// * `attn_cuda_type` - Type of CUDA attention.
///
/// # Returns
/// Result of the operation.
pub fn calc_deep_features_for_vamis(
    b_use_wand_for_image_loading: bool,
    variable_model_input_size: usize,
    window_size_pixels: usize,
    global_attn_indexes: &[usize],
    attn_cuda_type: usize,
) -> cResult<()> {
    let s_in_dataset_folder = format!("{}/10_images/PlantCLEF2025/PlantCLEF2025test/", D_DATASTORE);
    let d_in_dataset_folder = Path::new(&s_in_dataset_folder);

    let s_out_features_folder = if b_use_wand_for_image_loading {
        format!(
            "{}/20_deep_features/rust/PlantCLEF2025test_vamis{}_wsa3x3_backendwand/",
            D_DATASTORE, variable_model_input_size
        )
    } else {
        format!(
            "{}/20_deep_features/rust/PlantCLEF2025test_vamis{}_wsa3x3_backendimage/",
            D_DATASTORE, variable_model_input_size
        )
    };

    let d_out_features_folder = Path::new(&s_out_features_folder);

    lib_inference::calc_inferences_vamis_deep_features(
        d_in_dataset_folder,
        d_out_features_folder,
        true,
        true,
        variable_model_input_size,
        window_size_pixels,
        global_attn_indexes,
        attn_cuda_type,
        b_use_wand_for_image_loading,
    )
    .expect("Error in calc_inferences_vamis_deep_features");

    Ok(())
}
