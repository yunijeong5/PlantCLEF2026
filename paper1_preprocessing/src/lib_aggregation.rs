//! Module for analyzing tile-level detection aggregation at the plot scale.
//!
//! This module exports relevant data for:
//! - Qualitative analysis,
//! - Classical machine learning pipelines,
//!   to leverage all information provided by each plot's tiles.

use std::collections::HashSet;
use std::fs::File;
use std::io;
use std::path::Path;

use candle_core::{D, IndexOp, Result as cResult, Tensor};

use crate::lib_files::{
    get_basename, get_folder_tensors_list, load_tensor, read_text_file, save_tensor,
};
use crate::lib_inference::get_compute_device;
use crate::lib_model::{HeadModule, load_model_dinov2_base_pc24};
use crate::lib_soumission::SoumissionDataframe;

pub const DATASTORE: &str = "data";

#[allow(dead_code)]
pub enum TopSpeciesFilter<'a> {
    ImageRegex(&'a str),
    TopK(usize),
    NewScalesList(&'a [usize]),
    Threshold(f32),
}

pub struct TopSpeciesFilenames {
    pub f_tensor: String,
    pub f_images_list: String,
    pub f_classes_list_per_image: String,
    pub f_scales_list: String,
}

#[allow(clippy::ptr_arg)]
impl TopSpeciesFilenames {
    pub fn from_values(
        n_images: usize,
        scales_list: &Vec<usize>,
        n_top_classes: usize,
        tag: Option<String>,
    ) -> cResult<Self> {
        let prefix =
            String::from(DATASTORE) + "/50_probas_predictions_csv/rust_tiling_top_tensors/";
        let scales_string: String = scales_list.iter().map(|&s| s.to_string()).collect();

        let varfix = if let Some(tag_value) = tag {
            format!(
                "toptensor_img{}_scales{}_top{}_{}",
                n_images, scales_string, n_top_classes, tag_value
            )
        } else {
            format!(
                "toptensor_img{}_scales{}_top{}",
                n_images, scales_string, n_top_classes
            )
        };

        Ok(TopSpeciesFilenames {
            f_tensor: String::from(&prefix) + &varfix + ".safetensors",
            f_images_list: String::from(&prefix) + &varfix + "_images_list.csv",
            f_classes_list_per_image: String::from(&prefix)
                + &varfix
                + "_classes_lists_per_image.csv",
            f_scales_list: String::from(&prefix) + &varfix + "_scales_list.csv",
        })
    }
}

pub struct TopSpecies {
    pub top_tensor: Tensor,
    pub images_list: Vec<String>,
    pub species_list_per_image: Vec<Vec<String>>,
    pub scales_list: Vec<usize>,
}

#[allow(dead_code)]
impl TopSpecies {
    pub fn get_n_images(&self) -> usize {
        self.top_tensor.dims()[0]
    }

    pub fn get_n_tiles(&self) -> usize {
        self.top_tensor.dims()[1]
    }

    pub fn get_n_top_classes(&self) -> usize {
        self.top_tensor.dims()[2]
    }

    pub fn calc_top_k_for_image(
        tiles_probas: &Tensor,
        n_top_classes: usize,
    ) -> cResult<Vec<usize>> {
        let image_probas = tiles_probas.max(0).expect("Error calculating max");

        let prs: Vec<f32> = image_probas.to_vec1::<f32>().expect("Error");
        let mut prs = prs.iter().enumerate().collect::<Vec<_>>();
        prs.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));

        let class_index_list: Vec<usize> = prs
            .into_iter()
            .take(n_top_classes)
            .map(|(index, _value)| index)
            .collect();

        Ok(class_index_list)
    }

    /// Combines SoftMax probabilities (or logits) from all scales into a single tensor
    pub fn load_from_deep_features(
        available_scales: &[usize],
        n_top_classes: usize,
    ) -> cResult<Self> {
        // Données d'entrée, constantes
        let d_deep_features = String::from(DATASTORE) + "/20_deep_features/rust/";
        let f_class_list = String::from(DATASTORE) + "/30_models/species_id_mapping2.txt";
        let full_class_list: Vec<String> = read_text_file(&f_class_list).into_iter().collect();
        let n_classes: usize = full_class_list.len(); // 7806 - DINOv2Reg4 végétal PlantCLEF2024
        println!("Nombre de classes: {n_classes}");
        let n_tiles = available_scales.iter().fold(0, |acc, x| acc + x * x);
        let embed_dim = 768;
        let b_true_probas_false_logits_output = true; // Output des probas, pas des logits

        // Devices
        let use_cpu = false; // For compute device
        let compute_device = get_compute_device(use_cpu)?;
        let storage_device = candle_core::Device::Cpu;

        // Retrieve the list of all plot names into a vector
        // to count them and iterate over them multiple times
        let scale = available_scales[0];
        let d_deep_features_scale1 = String::from(&d_deep_features)
            + "PlantCLEF2025test_tiling"
            + &scale.to_string()
            + "x"
            + &scale.to_string()
            + "_backendwand";
        let p_deep_features_scale1 = Path::new(&d_deep_features_scale1);
        let images_iter = get_folder_tensors_list(p_deep_features_scale1).expect("Error");

        // Output structure
        //let mut output_image_list: Vec<String> = images_iter
        let output_image_list: Vec<String> = images_iter
            .map(|path_buf| path_buf.into_os_string().into_string().unwrap())
            .collect();
        let n_images = output_image_list.len();
        let mut output_species_list_per_image: Vec<Vec<String>> = Vec::new();
        // Allocate output tensor
        //let shape = &[n_images, n_tiles, n_top_classes];
        //let mut output_tensor = Tensor::zeros(shape, candle_core::DType::F32, &storage_device).expect("Error" );
        //let taille_mo = 4 * n_images * n_tiles * n_top_classes / 1024 / 1024;
        //println!("Tenseur initialisé, de dimensions: {:?}, taille en mémoire: {} Mo", output_tensor.shape(), taille_mo);
        let mut output_tensor_rows = Vec::new();

        let model = load_model_dinov2_base_pc24(&compute_device)?;

        // Main loop over images
        for (image_index, f_image) in output_image_list.iter().enumerate() {
            let f_image_short = get_basename(f_image).expect("Error");
            let image_index_inc = image_index + 1;
            println!("Traitement de l'image {image_index_inc} / {n_images} : {f_image_short}");

            // 1) Determine the best-detected species in the current plot (max pooling)

            // 1a) Retrieve the deep features of the current plot
            let shape = &[n_tiles, embed_dim];
            let mut tiles_df =
                Tensor::zeros(shape, candle_core::DType::F32, &compute_device).expect("Error");

            //let mut n_tiles_acc = 0;
            let n_tiles_acc = 0;
            //for (scale_index, scale) in available_scales.into_iter().enumerate() {
            for scale in available_scales.iter() {
                let f_tensor = String::from(&d_deep_features)
                    + "PlantCLEF2025test_tiling"
                    + &scale.to_string()
                    + "x"
                    + &scale.to_string()
                    + "_backendwand/"
                    + &f_image_short;
                let p_tensor = Path::new(&f_tensor);

                let tiles_df_cour = load_tensor(p_tensor, &compute_device).expect("Error");
                //println!("tiles_df_cour.shape: {:?}", tiles_df_cour.shape());

                let n_tiles_cour = scale * scale;

                let index_min = n_tiles_acc;
                let index_max = n_tiles_acc + n_tiles_cour;
                tiles_df = tiles_df
                    .slice_assign(&[index_min..index_max, 0..embed_dim], &tiles_df_cour)
                    .expect("N'a pu faire slice_assign");
            }

            // 1b) Apply the classification head to get logits, then probabilities
            let tiles_logits = model.forward_head(&tiles_df)?;
            let tiles_probas =
                //candle_nn::ops::softmax(&&tiles_logits.unsqueeze(0)?, D::Minus1)?.i(0)?;
                candle_nn::ops::softmax(&tiles_logits.unsqueeze(0)?, D::Minus1)?.i(0)?;

            // 1c) Determine the list of indices for the best-detected classes (TopK)
            let indices: Vec<usize> =
                TopSpecies::calc_top_k_for_image(&tiles_probas, n_top_classes).expect("Error");

            // 1d) Derive the list of classes and copy their probabilities/logits to the output tensor
            let class_list = indices
                .iter()
                .map(|i| full_class_list[*i].clone())
                .collect();
            output_species_list_per_image.push(class_list);

            let output_tensor_cour = if b_true_probas_false_logits_output {
                tiles_probas
            } else {
                tiles_logits
            };

            // Filter species list
            let mut rows = Vec::new();
            for i in indices {
                let row = output_tensor_cour.narrow(1, i, 1)?;
                rows.push(row);
            }
            let output_tensor_cour_bis = Tensor::cat(&rows, 1)?;

            let output_tensor_cour = output_tensor_cour_bis
                .unsqueeze(0)?
                .to_device(&storage_device)?;

            output_tensor_rows.push(output_tensor_cour);
        }

        println!("Building top tensor..");
        let output_tensor = Tensor::cat(&output_tensor_rows, 0)?;
        println!("Built");

        Ok(TopSpecies {
            top_tensor: output_tensor,
            images_list: output_image_list,
            species_list_per_image: output_species_list_per_image,
            scales_list: Vec::from(available_scales),
        })
    }

    pub fn save_to_files(&self, tag: Option<String>) -> cResult<()> {
        // impl TopSpeciesFilenames {
        // pub fn from_values(n_images: usize, scales_list: Vec<usize>, n_top_classes: usize) -> cResult<Self> {
        let filenames = TopSpeciesFilenames::from_values(
            self.get_n_images(),
            &self.scales_list,
            self.get_n_top_classes(),
            tag,
        )
        .expect("Error");

        // If one of the files exist, error
        if Path::new(&filenames.f_tensor).exists() {
            return Err(candle_core::Error::Io(io::Error::other(
                "Le fichier existe déjà.",
            )));
        }
        if Path::new(&filenames.f_images_list).exists() {
            return Err(candle_core::Error::Io(io::Error::other(
                "Le fichier existe déjà.",
            )));
        }
        if Path::new(&filenames.f_scales_list).exists() {
            return Err(candle_core::Error::Io(io::Error::other(
                "Le fichier existe déjà.",
            )));
        }
        if Path::new(&filenames.f_classes_list_per_image).exists() {
            return Err(candle_core::Error::Io(io::Error::other(
                "Le fichier existe déjà.",
            )));
        }

        // Save outputs
        save_tensor(&self.top_tensor, Path::new(&filenames.f_tensor));

        {
            let file_out_images_list = File::create(&filenames.f_images_list)?;
            let mut writer = csv::Writer::from_writer(file_out_images_list);
            //writer.write_record(&["image_name"]).expect("Error");
            writer.write_record(["image_name"]).expect("Error");
            for img in self.images_list.iter() {
                writer
                    //.write_record(&[get_basename(&img)
                    .write_record(&[get_basename(img)
                        .expect("Error")
                        .replace(".safetensors", "")])
                    .expect("Error");
            }
            writer.flush()?;
            println!("Fichier enregistré: {}", filenames.f_images_list);
        }

        {
            let n_top_classes = self.get_n_top_classes();
            let file_out_classes_list_per_image =
                File::create(&filenames.f_classes_list_per_image)?;
            let mut writer = csv::Writer::from_writer(file_out_classes_list_per_image);
            let headers: Vec<String> = (0..n_top_classes)
                .map(|i| format!("class{}", i + 1))
                .collect();
            writer.write_record(&headers).expect("Error");
            for classes in self.species_list_per_image.iter() {
                let mut record = Vec::new();
                for i in 0..n_top_classes {
                    if i < classes.len() {
                        record.push(classes[i].clone());
                    } else {
                        record.push("".to_string()); // Remplir avec une chaîne vide si la classe n'existe pas
                    }
                }
                writer.write_record(&record).expect("Error");
            }
            writer.flush()?;
            println!(
                "Fichier enregistré: {}",
                &filenames.f_classes_list_per_image
            );
        }

        {
            let file_out_scales_list = File::create(&filenames.f_scales_list)?;
            let mut writer = csv::Writer::from_writer(file_out_scales_list);
            //writer.write_record(&["scale"]).expect("Error");
            writer.write_record(["scale"]).expect("Error");
            for scale in self.scales_list.iter() {
                writer.write_record(&[scale.to_string()]).expect("Error");
            }
            writer.flush()?;
            println!("Fichier enregistré: {}", &filenames.f_scales_list);
        }

        Ok(())
    }

    /// Function to load the 'big' tensor (2105 plots x 145 tuiles x 100 espèces)
    pub fn load_from_files(
        n_images: usize,
        scales_list: &Vec<usize>,
        n_top_classes: usize,
        tag: Option<String>,
    ) -> cResult<Self> {
        let filenames = TopSpeciesFilenames::from_values(n_images, scales_list, n_top_classes, tag)
            .expect("Error");

        // Devices
        //let use_cpu = false; // For compute device
        //let compute_device = get_compute_device(use_cpu)?;
        let storage_device = candle_core::Device::Cpu;

        let p_in_tensor = Path::new(&filenames.f_tensor);
        let top_tensor = load_tensor(p_in_tensor, &storage_device).expect("Error");

        let (n_images, n_tiles, n_top_classes) = top_tensor.dims3()?;

        let images_list: Vec<String> = read_text_file(&filenames.f_images_list)
            .into_iter()
            .skip(1)
            .filter(|s| !s.is_empty())
            .collect();
        let n_images_check: usize = images_list.len(); // 7806 - DINOv2Reg4 végétal PlantCLEF2024
        //println!("images_list[-2] = {}", images_list[images_list.len() - 2]);
        //println!("images_list[-1] = {}", images_list[images_list.len() - 1]);
        assert_eq!(n_images, n_images_check);

        let scales_list: Vec<usize> = read_text_file(&filenames.f_scales_list)
            .into_iter()
            .skip(1)
            .filter(|s| !s.is_empty())
            .map(|s| {
                let scale: usize = s.parse().expect("Error");
                scale
            })
            .collect();
        // Check inputs
        let n_tiles_check = scales_list.iter().fold(0, |acc, x| acc + x * x);
        assert_eq!(n_tiles, n_tiles_check);

        let classes_list_as_string_per_image: Vec<String> =
            read_text_file(&filenames.f_classes_list_per_image)
                .into_iter()
                .skip(1)
                .filter(|s| !s.is_empty())
                .collect();
        let classes_list_per_image: Vec<Vec<String>> = classes_list_as_string_per_image
            .into_iter()
            .map(|s| s.split(',').map(|s| s.to_string()).collect())
            .collect();
        let n_images_check: usize = classes_list_per_image.len(); // 7806 - DINOv2Reg4 végétal PlantCLEF2024
        assert_eq!(n_images, n_images_check);

        for classes_list in classes_list_per_image.iter() {
            assert_eq!(classes_list.len(), n_top_classes);
        }

        // Filtrage, à la demande de l'utilisateur
        Ok(TopSpecies {
            top_tensor,
            images_list,
            species_list_per_image: classes_list_per_image,
            scales_list,
        })
    }

    pub fn apply_filter(&self, filter: TopSpeciesFilter) -> cResult<TopSpecies> {
        println!(
            "Entrée dans apply_filter. self.top_tensor.shape()={:?}",
            self.top_tensor.shape()
        );

        let storage_device = candle_core::Device::Cpu;

        // Build new tensor
        //let mut output_top_tensor = self.top_tensor.clone();
        //let output_images_list = self.images_list.clone();

        match filter {
            TopSpeciesFilter::ImageRegex(_regex) => {
                panic!("images filter with regex not implemented");
            }

            TopSpeciesFilter::TopK(k) => {
                // Vérifier que le new_n_top_classes < n_top_classes
                assert!(
                    k <= self.get_n_top_classes(),
                    "Le nouveau topK > topK actuel"
                );

                // Filtering by new_top_classes:
                //   * Recompute indices of the most probable species
                //   * Extract the new species list per plot, which must be included in the previous one
                //   * Build the new tensor
                //   let new_tensor = self.top_tensor.i(.., ..)
                let mut output_top_tensor = self.top_tensor.clone();
                let mut output_species_list_per_image = Vec::new();
                let n_top_classes = self.get_n_top_classes();
                if k != n_top_classes {
                    let mut output_top_tensor_list = Vec::new();
                    for (i, _s) in self.images_list.iter().enumerate() {
                        let tiles_probas: Tensor = self.top_tensor.i(i)?.clone();
                        //println!("debug: {:?}", tiles_probas.shape());
                        let indices =
                            TopSpecies::calc_top_k_for_image(&tiles_probas, k).expect("Error");

                        let class_list: Vec<String> = indices
                            .iter()
                            .map(|j| self.species_list_per_image[i][*j].clone())
                            .collect();
                        output_species_list_per_image.push(class_list);

                        // filtrer la liste des probas des espèces
                        let mut rows = Vec::new();
                        for j in indices {
                            let row = tiles_probas.narrow(1, j, 1)?;
                            rows.push(row);
                        }
                        let output_tensor_cour = Tensor::cat(&rows, 1)?
                            .unsqueeze(0)?
                            .to_device(&storage_device)?;
                        output_top_tensor_list.push(output_tensor_cour);
                    }
                    output_top_tensor = Tensor::cat(&output_top_tensor_list, 0).expect("Error");
                }

                Ok(TopSpecies {
                    top_tensor: output_top_tensor,
                    images_list: self.images_list.clone(),
                    species_list_per_image: output_species_list_per_image,
                    scales_list: self.scales_list.clone(),
                })
            }

            TopSpeciesFilter::NewScalesList(new_scales_list) => {
                // Ensure the new scales are included in the previous ones
                let set: HashSet<_> = self.scales_list.iter().collect();
                assert!(
                    new_scales_list.iter().all(|item| set.contains(item)),
                    "Erreur: Certaines échelles demandées ne sont pas dans la liste actuelle"
                );

                // Filter through Scales
                let mut new_tensor_cols = Vec::new();
                let mut tile_index_cour = 0;
                for scale in self.scales_list.clone() {
                    let n_tiles = scale * scale;
                    if new_scales_list.contains(&scale) {
                        let cols = self.top_tensor.narrow(1, tile_index_cour, n_tiles)?.clone();
                        new_tensor_cols.push(cols);
                    }
                    tile_index_cour += n_tiles;
                }
                let output_top_tensor = Tensor::cat(&new_tensor_cols, 1).expect("Error");
                let output_scales_list = new_scales_list;

                Ok(TopSpecies {
                    top_tensor: output_top_tensor,
                    images_list: self.images_list.clone(),
                    species_list_per_image: self.species_list_per_image.clone(),
                    scales_list: output_scales_list.to_vec(),
                })
            }

            TopSpeciesFilter::Threshold(threshold) => {
                // Filter using the threshold (only affects the Vec<String> of species lists per image)
                let max_tiles = self.top_tensor.max(1).expect("Error").clone();
                let output_species_list_per_image: Vec<Vec<String>> = self
                    .images_list
                    .iter()
                    .enumerate()
                    .map(|(i, _s)| {
                        max_tiles
                            .i(i)
                            .expect("Error")
                            .to_vec1::<f32>()
                            .expect("Error")
                            .iter()
                            .enumerate()
                            //.filter(|(j, v)| v.clone() >= &threshold)
                            .filter(|(_j, v)| v >= &&threshold)
                            .map(|(j, _v)| self.species_list_per_image[i][j].clone())
                            .collect()
                    })
                    .collect();
                println!(
                    "Warning: Dans TopSpeciesFilter::Threshold(), le tenseur top n'est pas mis à jour. Valable seulement pour calculer une soumission sans autre filtre"
                );

                Ok(TopSpecies {
                    top_tensor: self.top_tensor.clone(),
                    images_list: self.images_list.clone(),
                    species_list_per_image: output_species_list_per_image,
                    scales_list: self.scales_list.clone(),
                })
            }
        }
    }

    /// Function to compute the submission
    pub fn save_as_soumission(&self, f_out_soumission: &str) {
        let lines: Vec<(String, Vec<String>)> = self
            .images_list
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), self.species_list_per_image[i].clone()))
            .collect();
        let soumission = SoumissionDataframe { lines };
        let _ = soumission.to_csv(f_out_soumission);
    }
}
