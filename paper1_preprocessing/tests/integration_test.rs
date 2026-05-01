// tests/integration_test.rs

use plantclef2025_rs::lib_files::load_csv_to_hashmap;
use plantclef2025_rs::{
    calc_inferences_518_predictions_et_soumission, inference_single_image, main_calc_tiling,
};
use std::path::Path; //, PathBuf}; // Remplace `multilabel` par le nom de ton crate

pub const D_DATASTORE: &str = "data";

#[test]
fn test_inference_simple_image_1_518() {
    println!("test_inference_simple_image (1): Single inference 518x518:");
    // Paramètres communs à tous les tests
    // Chemin vers l'image de test
    let f_img_in = "data/10_images/valeriana_tuberosa_small.jpeg";

    // Chemin vers le fichier de mapping des espèces
    let f_species_id_mapping = "data/30_models/species_id_mapping.csv";
    let b_use_cpu_for_computations = false;
    let b_use_pc24model = true;
    let attn_cuda_type = 3; // TODO: Indiquer chaque type d'attention

    let model_input_size = 518;
    let b_use_pc24wsa = false;
    // Not used
    let window_size = 0; // 518pixels / 14pixel/patch = 37 patchs
    let global_attn_indexes = &[];

    let (top_species, top_probs) = inference_single_image(
        f_img_in,
        b_use_cpu_for_computations,
        b_use_pc24model,
        model_input_size,
        b_use_pc24wsa,
        window_size,
        global_attn_indexes,
        attn_cuda_type,
        f_species_id_mapping,
    )
    .unwrap();
    println!("top_species: {:?}", top_species);
    println!("top_probs: {:?}", top_probs);

    // Vérification des résultats
    assert_eq!(top_species[0], "Valeriana tuberosa L.");
    assert_eq!(top_species[1], "Valeriana apula Pourr.");
    assert_eq!(
        top_species[2],
        "Valeriana lecoqii (Jord.) Christenh. & Byng"
    );
    assert_eq!(top_species[3], "Valeriana montana L.");
    assert_eq!(top_species[4], "Valeriana angustifolia Mill.");
    // Vérification des probabilités (avec une tolérance pour les floats)
    assert!((top_probs[0] - 0.7852).abs() < 0.01);
    assert!((top_probs[1] - 0.0140).abs() < 0.01);
    assert!((top_probs[2] - 0.0050).abs() < 0.01);
    assert!((top_probs[3] - 0.0049).abs() < 0.01);
    assert!((top_probs[4] - 0.0039).abs() < 0.01);
}

#[test]
fn test_inference_simple_image_2_vamis() {
    println!("test_inference_simple_image (2): Single inference 1554x1554 (classic VaMIS):");
    // Paramètres communs à tous les tests
    // Chemin vers l'image de test
    let f_img_in = "data/10_images/valeriana_tuberosa_small.jpeg";

    // Chemin vers le fichier de mapping des espèces
    let f_species_id_mapping = "data/30_models/species_id_mapping.csv";
    let b_use_cpu_for_computations = false;
    let b_use_pc24model = true;
    let attn_cuda_type = 3; // TODO: Indiquer chaque type d'attention

    let model_input_size = 1554;
    let b_use_pc24wsa = false;
    // Not used
    let window_size = 0; // 518pixels / 14pixel/patch = 37 patchs
    let global_attn_indexes = &[]; // Global Attention like SegmentAnything

    let (top_species, top_probs) = inference_single_image(
        f_img_in,
        b_use_cpu_for_computations,
        b_use_pc24model,
        model_input_size,
        b_use_pc24wsa,
        window_size,
        global_attn_indexes,
        attn_cuda_type,
        f_species_id_mapping,
    )
    .unwrap();
    println!("top_species: {:?}", top_species);
    println!("top_probs: {:?}", top_probs);

    // Vérification des résultats
    assert_eq!(top_species[0], "Valeriana tuberosa L.");
    assert_eq!(top_species[1], "Valeriana apula Pourr.");
    assert_eq!(top_species[2], "Valeriana saliunca All.");
    assert_eq!(top_species[3], "Valeriana trinervis Viv.");
    assert_eq!(top_species[4], "Valeriana montana L.");
    // Vérification des probabilités (avec une tolérance pour les floats)
    assert!((top_probs[0] - 0.8965).abs() < 0.01);
    assert!((top_probs[1] - 0.0132).abs() < 0.01);
    assert!((top_probs[2] - 0.0036).abs() < 0.01);
    assert!((top_probs[3] - 0.0026).abs() < 0.01);
    assert!((top_probs[4] - 0.0018).abs() < 0.01);
}

// test is desactivated as too long
//#[test]
#[allow(dead_code)]
fn test_inference_simple_image_3_vamis_via_wsa() {
    println!(
        "test_inference_simple_image (3): Single inference 1554x1554 (classic VaMIS) - via WindowShiftedAttention code):"
    );
    // Paramètres communs à tous les tests
    // Chemin vers l'image de test
    let f_img_in = "data/10_images/valeriana_tuberosa_small.jpeg";

    // Chemin vers le fichier de mapping des espèces
    let f_species_id_mapping = "data/30_models/species_id_mapping.csv";
    let b_use_cpu_for_computations = false;
    let b_use_pc24model = true;
    let attn_cuda_type = 3; // TODO: Indiquer chaque type d'attention

    let model_input_size = 1554;
    let b_use_pc24wsa = true;
    // Not used
    let window_size = 37; // 518pixels / 14pixel/patch = 37 patchs
    let global_attn_indexes = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]; // Global Attention only

    let (top_species, top_probs) = inference_single_image(
        f_img_in,
        b_use_cpu_for_computations,
        b_use_pc24model,
        model_input_size,
        b_use_pc24wsa,
        window_size,
        global_attn_indexes,
        attn_cuda_type,
        f_species_id_mapping,
    )
    .unwrap();
    println!("top_species: {:?}", top_species);
    println!("top_probs: {:?}", top_probs);

    // Vérification des résultats
    assert_eq!(top_species[0], "Valeriana tuberosa L.");
    assert_eq!(top_species[1], "Valeriana apula Pourr.");
    assert_eq!(top_species[2], "Valeriana saliunca All.");
    assert_eq!(top_species[3], "Valeriana trinervis Viv.");
    assert_eq!(top_species[4], "Valeriana montana L.");
    // Vérification des probabilités (avec une tolérance pour les floats)
    assert!((top_probs[0] - 0.8965).abs() < 0.01);
    assert!((top_probs[1] - 0.0132).abs() < 0.01);
    assert!((top_probs[2] - 0.0036).abs() < 0.01);
    assert!((top_probs[3] - 0.0026).abs() < 0.01);
    assert!((top_probs[4] - 0.0018).abs() < 0.01);
}

#[test]
fn test_inference_simple_image_4_vamis_wsa3x3() {
    println!(
        "test_inference_simple_image (4): Single inference 1554x1554 (VaMIS WindowShiftedAttention 3x3):"
    );
    // Paramètres communs à tous les tests
    // Chemin vers l'image de test
    let f_img_in = "data/10_images/valeriana_tuberosa_small.jpeg";

    // Chemin vers le fichier de mapping des espèces
    let f_species_id_mapping = "data/30_models/species_id_mapping.csv";
    let b_use_cpu_for_computations = false;
    let b_use_pc24model = true;
    let attn_cuda_type = 3; // TODO: Indiquer chaque type d'attention

    let model_input_size = 1554;
    let b_use_pc24wsa = true;
    let window_size = 37; // 1554 / (14pixels/patch * ws=37) = 1554 / 518 = 3  ->  3x3 window partition
    let global_attn_indexes = &[]; // Only Local Attention

    let (top_species, top_probs) = inference_single_image(
        f_img_in,
        b_use_cpu_for_computations,
        b_use_pc24model,
        model_input_size,
        b_use_pc24wsa,
        window_size,
        global_attn_indexes,
        attn_cuda_type,
        f_species_id_mapping,
    )
    .unwrap();
    println!("top_species: {:?}", top_species);
    println!("top_probs: {:?}", top_probs);

    // Vérification des résultats
    assert_eq!(top_species[0], "Valeriana tuberosa L.");
    assert_eq!(top_species[1], "Valeriana apula Pourr.");
    assert_eq!(top_species[2], "Valeriana officinalis L.");
    assert_eq!(top_species[3], "Valeriana angustifolia Mill.");
    assert_eq!(top_species[4], "Valeriana calcitrapae L.");
    // Vérification des probabilités (avec une tolérance pour les floats)
    assert!((top_probs[0] - 0.9428).abs() < 0.01);
    assert!((top_probs[1] - 0.0137).abs() < 0.01);
    assert!((top_probs[2] - 0.0025).abs() < 0.01);
    assert!((top_probs[3] - 0.0022).abs() < 0.01);
    assert!((top_probs[4] - 0.0020).abs() < 0.01);
}

#[test]
fn test_inference_simple_image_5_vamis_attnlikesam() {
    println!(
        "test_inference_simple_image (5): Single inference 1554x1554 (VaMIS Global attn like SAM 3x3):"
    );
    // Paramètres communs à tous les tests
    // Chemin vers l'image de test
    let f_img_in = "data/10_images/valeriana_tuberosa_small.jpeg";

    // Chemin vers le fichier de mapping des espèces
    let f_species_id_mapping = "data/30_models/species_id_mapping.csv";
    let b_use_cpu_for_computations = false;
    let b_use_pc24model = true;
    let attn_cuda_type = 3; // TODO: Indiquer chaque type d'attention

    let model_input_size = 1554;
    let b_use_pc24wsa = true;
    let window_size = 37; // 1554 / (14pixels/patch * ws=37) = 1554 / 518 = 3  ->  3x3 window partition
    // Global Attention like SegmentAnything (after ViT blocs 3,6,9,12):
    let global_attn_indexes = &[2, 5, 8, 11];

    let (top_species, top_probs) = inference_single_image(
        f_img_in,
        b_use_cpu_for_computations,
        b_use_pc24model,
        model_input_size,
        b_use_pc24wsa,
        window_size,
        global_attn_indexes,
        attn_cuda_type,
        f_species_id_mapping,
    )
    .unwrap();
    println!("top_species: {:?}", top_species);
    println!("top_probs: {:?}", top_probs);

    // Vérification des résultats
    assert_eq!(top_species[0], "Valeriana tuberosa L.");
    assert_eq!(top_species[1], "Valeriana apula Pourr.");
    assert_eq!(top_species[2], "Valeriana saliunca All.");
    assert_eq!(top_species[3], "Valeriana trinervis Viv.");
    assert_eq!(top_species[4], "Valeriana angustifolia Mill.");

    // Vérification des probabilités (avec une tolérance pour les floats)
    assert!((top_probs[0] - 0.9908).abs() < 0.01);
    assert!((top_probs[1] - 0.0055).abs() < 0.01);
    assert!((top_probs[2] - 0.0004).abs() < 0.01);
    assert!((top_probs[3] - 0.0002).abs() < 0.01);
    assert!((top_probs[4] - 0.0002).abs() < 0.01);
}

#[allow(clippy::too_many_arguments)]
fn calc_inferences_518_predictions_et_soumission_internal(
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
) {
    let expected_f_out_csv_predictions_short =
        format!("{}_{}_predictions_{}.csv", tag_dataset, tag_date, tag_sup);
    let expected_f_out_csv_submission_short =
        format!("{}_{}_submission_{}.csv", tag_dataset, tag_date, tag_sup);

    let expected_f_out_csv_predictions = format!(
        "{}/50_probas_predictions_csv/{}",
        D_DATASTORE, expected_f_out_csv_predictions_short
    );
    let expected_f_out_csv_submission = format!(
        "{}/65_submissions/{}",
        D_DATASTORE, expected_f_out_csv_submission_short
    );

    // Chemins pour les fichiers de référence (même basename)
    let f_reference_predictions =
        format!("tests/references/{}", expected_f_out_csv_predictions_short);
    let f_reference_submission =
        format!("tests/references/{}", expected_f_out_csv_submission_short);

    // Exécution de la fonction
    let result = calc_inferences_518_predictions_et_soumission(
        tag_dataset,
        d_sub_dataset_folder,
        tag_sup,
        tag_date,
        batch_size,
        model_input_size,
        top_k,
        min_score,
        b_write_csv_submission,
        use_cpu_for_computations,
    );

    // Vérification que la fonction s'est exécutée sans erreur
    assert!(result.is_ok(), "La fonction doit réussir sans erreur");

    // Vérification que les fichiers de sortie existent
    assert!(
        Path::new(&expected_f_out_csv_predictions).exists(),
        "Le fichier de prédictions doit exister : {}",
        expected_f_out_csv_predictions
    );

    // Vérification du fichier de soumission (si activé)
    if b_write_csv_submission {
        assert!(
            Path::new(&expected_f_out_csv_submission).exists(),
            "Le fichier de soumission doit exister : {}",
            expected_f_out_csv_submission
        );
    }

    // Comparaison du fichier de prédictions avec la référence
    //if Path::new(&f_reference_predictions).exists() {
    // Charge les prédictions générées et la référence en HashMap
    let predictions_map = load_csv_to_hashmap(
        &expected_f_out_csv_predictions,
        "image_name",
        "1741661",
        None,
    )
    .expect("Échec du chargement des prédictions générées");
    let reference_map =
        load_csv_to_hashmap(&f_reference_predictions, "image_name", "1741661", None)
            .expect("Échec du chargement des prédictions de référence");

    // Vérifie que les deux HashMaps sont identiques
    assert_eq!(
        predictions_map, reference_map,
        "Les prédictions doivent correspondre à la référence"
    );
    //}

    // Comparaison du fichier de soumission avec la référence (si disponible)
    if b_write_csv_submission {
        // && Path::new(&f_reference_submission).exists() {
        let submission_map = load_csv_to_hashmap(
            &expected_f_out_csv_submission,
            "quadrat_id",
            "species_ids",
            Some(b','),
        )
        .expect("Échec du chargement de la soumission générée");
        let reference_submission_map = load_csv_to_hashmap(
            &f_reference_submission,
            "quadrat_id",
            "species_ids",
            Some(b','),
        )
        .expect("Échec du chargement de la soumission de référence");

        // Vérifie que les deux HashMaps sont identiques
        assert_eq!(
            submission_map, reference_submission_map,
            "La soumission doit correspondre à la référence"
        );
    }
}

#[test]
fn test_calc_inferences_518_predictions_et_soumission() {
    // Paramètres du test
    let tag_dataset = "PlantCLEF2025test_5";
    //let tag_dataset = "PlantCLEF2025test";
    let d_sub_dataset_folder = "PlantCLEF2025";
    let tag_sup = "test_baseline518";
    let tag_date = "2026-01-03"; // Format AAAAMMJJ
    //let batch_size = 5;
    let batch_size = 64;
    let model_input_size = 518;
    let top_k = 15;
    let min_score = 0.01;
    let b_write_csv_submission = true;
    let use_cpu_for_computations = false;
    //let use_cpu_for_computations = true;

    calc_inferences_518_predictions_et_soumission_internal(
        tag_dataset,
        d_sub_dataset_folder,
        tag_sup,
        tag_date,
        batch_size,
        model_input_size,
        top_k,
        min_score,
        b_write_csv_submission,
        use_cpu_for_computations,
    );
}

fn test_main_calc_tiling_internal(
    tag_dataset: &str,
    tag_date: &str,
    scale_list: Vec<usize>,
    b_load_img_with_wand: bool,
    top_k: usize,
    min_score: f32,
) {
    // Chemins attendus pour les fichiers de sortie
    let n_tiles: usize = scale_list.iter().map(|&x| x * x).sum();
    let tag_df = if b_load_img_with_wand {
        format!("{}_tiling{}_backendwand", tag_dataset, &n_tiles)
    } else {
        format!("{}_tiling{}_backendimage", tag_dataset, &n_tiles)
    };

    let expected_f_out_csv_predictions_short =
        format!("{}_{}_rust_predictions.csv", tag_df, tag_date);
    let expected_f_out_csv_submission_short = format!(
        "{}_{}_submission_seuil_{:02.1}prcts.csv",
        tag_df,
        tag_date,
        min_score * 100.
    );

    let expected_f_out_csv_predictions = format!(
        "{}/50_probas_predictions_csv/{}",
        D_DATASTORE, expected_f_out_csv_predictions_short
    );
    let expected_f_out_csv_submission = format!(
        "{}/65_submissions/{}",
        D_DATASTORE, expected_f_out_csv_submission_short
    );
    // Chargement des fichiers de référence (si disponibles)
    let f_reference_predictions =
        format!("tests/references/{}", expected_f_out_csv_predictions_short);
    let f_reference_submission =
        format!("tests/references/{}", expected_f_out_csv_submission_short);

    // Exécution de la fonction
    let result = main_calc_tiling(
        tag_dataset,
        tag_date,
        scale_list,
        b_load_img_with_wand,
        top_k,
        min_score,
    );

    // Vérification que la fonction s'est exécutée sans erreur
    assert!(result.is_ok(), "La fonction doit réussir sans erreur");

    // Vérification que les fichiers de sortie existent
    assert!(
        Path::new(&expected_f_out_csv_predictions).exists(),
        "Le fichier de prédictions doit exister : {}",
        expected_f_out_csv_predictions
    );

    // Vérification du fichier de soumission (si activé)
    assert!(
        Path::new(&expected_f_out_csv_submission).exists(),
        "Le fichier de soumission doit exister : {}",
        expected_f_out_csv_submission
    );

    // Comparaison du fichier de prédictions avec la référence
    //if Path::new(&f_reference_predictions).exists() {
    // Charge les prédictions générées et la référence en HashMap
    let predictions_map = load_csv_to_hashmap(
        &expected_f_out_csv_predictions,
        "image_name",
        "1741661",
        None,
    )
    .expect("Échec du chargement des prédictions générées");
    let reference_map =
        load_csv_to_hashmap(&f_reference_predictions, "image_name", "1741661", None)
            .expect("Échec du chargement des prédictions de référence");

    // Vérifie que les deux HashMaps sont identiques
    assert_eq!(
        predictions_map, reference_map,
        "Les prédictions doivent correspondre à la référence"
    );
    //}

    // Comparaison du fichier de soumission avec la référence (si disponible)
    //if Path::new(&f_reference_submission).exists() {
    let submission_map = load_csv_to_hashmap(
        &expected_f_out_csv_submission,
        "quadrat_id",
        "species_ids",
        Some(b','),
    )
    .expect("Échec du chargement de la soumission générée");
    let reference_submission_map = load_csv_to_hashmap(
        &f_reference_submission,
        "quadrat_id",
        "species_ids",
        Some(b','),
    )
    .expect("Échec du chargement de la soumission de référence");

    // Vérifie que les deux HashMaps sont identiques
    assert_eq!(
        submission_map, reference_submission_map,
        "La soumission doit correspondre à la référence"
    );
    //}
}

#[test]
fn test_main_calc_tiling_1_only_one_tile() {
    let tag_dataset = "PlantCLEF2025test_5";
    //let tag_dataset = "PlantCLEF2025test";
    let b_load_img_with_wand = true;
    let top_k = 15;
    let tag_date = "2026-01-03";

    let scale_list = vec![1];
    let min_score = 0.01;
    test_main_calc_tiling_internal(
        tag_dataset,
        tag_date,
        scale_list,
        b_load_img_with_wand,
        top_k,
        min_score,
    );
}

#[test]
fn test_main_calc_tiling_2_10_tiles_on_2_scales() {
    let tag_dataset = "PlantCLEF2025test_5";
    //let tag_dataset = "PlantCLEF2025test";
    let b_load_img_with_wand = true;
    let top_k = 15;
    let tag_date = "2026-01-03";

    let scale_list = vec![1, 3]; // Scales 1 and 3 -> 1 + 3x3 = 10 tiles per plot
    let min_score = 0.10; // Seuil à 10 pourcents avec tiling 1+3x3
    test_main_calc_tiling_internal(
        tag_dataset,
        tag_date,
        scale_list,
        b_load_img_with_wand,
        top_k,
        min_score,
    );
}
