//! Module for analyzing and converting predictions into submission files for **PlantCLEF 2025**.
//!
//! **Features**:
//! - **Dichotomy**: Automatic threshold search to target an average number of species per image.
//! - **Filtering**: Species selection by threshold or top-K (with softmax option).
//! - **CSV Export**: Generation of submission files in the expected format (e.g., `quadrat_id,species_ids`).

use std::error::Error;
use std::fs::File;
use std::path::Path;

use csv::{QuoteStyle, WriterBuilder};

use candle_core::{D, Device, IndexOp, Result as cResult};

use crate::lib_files::PredictionsDataframe;

/// Dichotomy on an increasing function.
/// Recursively searches for the zero of a function using bisection.
/// The input evaluation function must be monotonically increasing.
pub fn dichotomy<F>(
    min_cour: f32,
    max_cour: f32,
    n_iter: usize,
    f_eval: F,
    went_up: bool,
    went_down: bool,
) -> Result<f32, Box<dyn Error>>
where
    F: Fn(f32) -> f32,
{
    if n_iter == 0 {
        if !went_up {
            return Err("Dichotomy did not converge. Try decrease min_cour.".into());
        }
        if !went_down {
            return Err("Dichotomy did not converge. Try increase max_cour.".into());
        }
        let mid_cour = 0.5 * (min_cour + max_cour);
        return Ok(mid_cour);
    }

    let mid_cour = 0.5 * (min_cour + max_cour);
    let f_mid_cour = f_eval(mid_cour);
    if f_mid_cour >= 0.0 {
        dichotomy(min_cour, mid_cour, n_iter - 1, f_eval, went_up, true)
    } else {
        dichotomy(mid_cour, max_cour, n_iter - 1, f_eval, true, went_down)
    }
}

/// Get the minimum value from an iterator
pub fn get_min_iter<I, J>(mut v: I) -> Option<J>
where
    J: PartialOrd + Copy,
    I: Iterator<Item = J>,
{
    // Use next to get the first element
    let first = v.next()?;

    // Use fold to find the minimum
    let min = v.fold(first, |min, x| if x < min { x } else { min });

    Some(min)
}

/// Get the maximum value from an iterator
pub fn get_max_iter<I, J>(mut v: I) -> Option<J>
where
    J: PartialOrd + Copy,
    I: Iterator<Item = J>,
{
    // Use next to get the first element
    let first = v.next()?;

    // Use fold to find the maximum
    let max = v.fold(first, |max, x| if x > max { x } else { max });

    Some(max)
}

/// Calculates the average number of species per image given a PredictionsDataframe and a threshold.
pub fn calc_nb_moy_species(df: &PredictionsDataframe, threshold: f32) -> f32 {
    let n_images = df.lines.len();
    let sum: usize = df
        .lines
        .iter()
        .map(|(_s, t)| {
            t.to_vec1::<f32>()
                .expect("Error")
                .iter()
                .filter(|&&value| value > threshold)
                .count()
        })
        .sum();
    sum as f32 / n_images as f32
}

/// Finds the optimal threshold to reach the target average number of species per image using dichotomy.
pub fn calc_seuil(
    df: &PredictionsDataframe,
    target_nb_species: f32,
    dicho_n_iter: usize,
) -> Result<f32, Box<dyn Error>> {
    let min_start = get_min_iter(
        df.lines
            .iter()
            .filter_map(|(_s, t)| get_min_iter(t.to_vec1::<f32>().expect("Error").into_iter())),
    )
    .expect("Pas de min car pas de predictions");

    let max_start = get_max_iter(
        df.lines
            .iter()
            .filter_map(|(_s, t)| get_max_iter(t.to_vec1::<f32>().expect("Error").into_iter())),
    )
    .expect("Pas de min car pas de predictions");

    let dicho_res = dichotomy(
        min_start,
        max_start,
        dicho_n_iter,
        |x| target_nb_species - calc_nb_moy_species(df, x),
        false,
        false,
    );

    println!("Seuil trouvé par dichotomie: {dicho_res:?}");

    match dicho_res {
        Ok(seuil) => Ok(seuil),
        Err(e) => Err(e),
    }
}

/// Represents a submission dataframe for the **PlantCLEF 2025** competition.
/// Contains a list of image IDs and their associated predicted species.
pub struct SoumissionDataframe {
    pub lines: Vec<(String, Vec<String>)>,
}

impl SoumissionDataframe {
    /// Converts a PredictionsDataframe into a SoumissionDataframe using a threshold.
    pub fn new_with_threshold(df: &PredictionsDataframe, threshold: f32) -> Self {
        let lines: Vec<(String, Vec<String>)> = df
            .lines
            .iter()
            .map(|(s, t)| {
                (
                    s.replace(".jpg", "").clone(),
                    t.to_vec1::<f32>()
                        .expect("Error")
                        .iter()
                        .enumerate()
                        .filter(|&(_index, &value)| value > threshold)
                        .map(|(index, _value)| df.class_list[index].clone())
                        .collect(),
                )
            })
            .collect();

        SoumissionDataframe { lines }
    }

    pub fn new_with_topk_and_min_score(
        df: &PredictionsDataframe,
        top_k: usize,
        min_score: f32,
        b_apply_softmax: bool,
    ) -> Self {
        //Predictions:: pub lines: Vec<(String, Tensor)>,
        let lines: Vec<(String, Vec<String>)> = df
            .lines
            .iter()
            .map(|(s, scores)| {
                let prs: Vec<f32> = if b_apply_softmax {
                    candle_nn::ops::softmax(&scores.unsqueeze(0).expect("Error"), D::Minus1)
                        .expect("Error")
                        .i(0)
                        .expect("Error")
                        .to_vec1::<f32>()
                        .expect("Error")
                } else {
                    scores.to_vec1::<f32>().expect("Error")
                };
                let mut prs = prs.iter().enumerate().collect::<Vec<_>>();
                prs.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));
                let class_list: Vec<String> = prs
                    .into_iter()
                    .take(top_k)
                    .filter(|&(_index, &value)| value > min_score)
                    .map(|(index, _value)| df.class_list[index].clone())
                    .collect();
                (
                    s.replace(".jpg", "").replace(".safetensors", "").clone(),
                    class_list,
                )
            })
            .collect();

        SoumissionDataframe { lines }
    }

    pub fn to_csv(&self, s_file_path: &str) -> cResult<()> {
        println!("Appel à SoumissionDataframe.to_csv()");
        let file_path = Path::new(s_file_path);
        let file = File::create(file_path)?;
        let mut wtr = WriterBuilder::new()
            //.delimiter(b';')
            .delimiter(b',')
            .quote_style(QuoteStyle::Never) // Add manually delimiters
            .from_writer(file);

        //let header = ["plot_id", "species_ids"];
        let header = ["quadrat_id", "species_ids"];
        wtr.write_record(header).expect("csv write error");

        // Écrire les lignes
        for (label, vec_usize) in self.lines.iter() {
            let s_vec_usize = format!("{:?}", vec_usize).replace("\"", "");
            let record = [format!("\"{}\"", label), format!("\"{}\"", s_vec_usize)];
            wtr.write_record(record).expect("csv write error");
        }

        wtr.flush()?;
        println!("Fichier écrit: {s_file_path:?}");
        Ok(())
    }
}

/// Generates a submission file for PlantCLEF 2025 from raw prediction data.
/// Applies filtering by top-K and/or minimum score, with optional softmax normalization.
/// Writes the results to a CSV file in the required submission format.
pub fn calc_submission_file_from_predictions_file(
    f_in_csv_predictions: &str,
    f_out_csv_submission: &str,
    top_k: usize,
    min_score: f32,
    b_apply_softmax: bool,
    storage_device: &Device,
) -> cResult<()> {
    // Chargement des prédictions par plot, avec les tenseurs vers le storage_device.
    println!("Chargement du fichier de prédictions par plot:{f_in_csv_predictions}");
    let df_predictions =
        //PredictionsDataframe::from_csv(f_in_csv_predictions, &storage_device).expect("Error");
        PredictionsDataframe::from_csv(f_in_csv_predictions, storage_device).expect("Error");

    if false {
        // Calcul de la soumission
        println!("Calcul du seuil de la soumission..");
        let target_nb_species = 10.0; // A choisir
        let dicho_n_iter = 20;
        let threshold = calc_seuil(&df_predictions, target_nb_species, dicho_n_iter)
            .expect("Erreur dans le calcul du seuil");
        println!("Seuil choisi: {threshold}");

        println!("Calcul de la soumission..");
        let _soumission = SoumissionDataframe::new_with_threshold(&df_predictions, threshold);
    }

    let soumission = SoumissionDataframe::new_with_topk_and_min_score(
        &df_predictions,
        top_k,
        min_score,
        b_apply_softmax,
    );
    soumission
        .to_csv(f_out_csv_submission)
        .expect("Erreur dans l'écriture du fichier soumission");

    Ok(())
}
