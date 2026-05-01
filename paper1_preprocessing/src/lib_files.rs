//! Utility module for managing files, images, and tensors in the context of **PlantCLEF 2025**.
//!
//! This module provides functions and structures for:
//! - **Listing and filtering** files in a directory (images, tensors).
//! - **Loading and resizing** images (with support for Wand/ImageMagick lib and Lanczos interpolation).
//! - **Manipulating tensors** (loading/saving in safetensors format).
//! - **Reading/writing tabular data** (CSV) for predictions.

use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Once;

use candle_core::safetensors::load as candle_safetensors_load;
use candle_core::{Device, Tensor};
use csv::{ReaderBuilder, WriterBuilder};
use image::{ImageBuffer, ImageReader, Rgb};

use magick_rust::{MagickWand, magick_wand_genesis};
static START: Once = Once::new();

pub const IMG_EXTENSIONS: [&str; 4] = ["jpg", "jpeg", "png", "bmp"];

pub const SAFETENSORS_EXTENSIONS: [&str; 2] = ["pth", "safetensors"];

pub fn get_folder_files_list(
    folder_path: &Path,
    valid_extensions: &[&str],
) -> Result<impl Iterator<Item = PathBuf>, Box<dyn Error>> {
    let mut files: Vec<PathBuf> = fs::read_dir(folder_path)
        .expect("Unable to read directory")
        .filter(move |entry| {
            let entry = entry.as_ref().expect("Unable to read directory entry");
            let path = entry.path();
            if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
                let extension_lower = extension.to_lowercase();
                valid_extensions.contains(&extension_lower.as_str())
            } else {
                false
            }
        })
        .map(|entry| entry.map(|entry| entry.path()).expect("error"))
        .collect();

    files.sort(); // Sort filenames

    Ok(files.into_iter())
}

pub fn get_folder_images_list(
    folder_path: &Path,
) -> Result<impl Iterator<Item = PathBuf>, Box<dyn Error>> {
    get_folder_files_list(folder_path, &IMG_EXTENSIONS)
}

/// Lightweight struct representing an image in raw bitmap format.
pub struct ImgVecU8 {
    pub data: Vec<u8>,
    pub height: usize,
    pub width: usize,
}

/*
Avoid turbojpeg dependancy

pub fn load_image_and_resize_with_turbojpeg_to518(f_in: &str) -> Result<ImgVecU8, Box<dyn Error>> {
    // Source: https://github.com/honzasp/rust-turbojpeg/blob/master/examples/decompressor.rs
    use turbojpeg::{Decompressor, Image, PixelFormat, Subsample, TJPF, TJSAMP, TJTransform};

    // get the JPEG data
    let jpeg_data = std::fs::read(f_in)?;

    // initialize a Decompressor
    let mut decompressor = Decompressor::new()?;

    // read the JPEG header with image size
    let header = decompressor.read_header(&jpeg_data)?;
    //let (width, height) = (header.width, header.height);
    let width = 518;
    let height = 518;

    // prepare the destination image
    let mut img = Image {
        pixels: vec![0; 3 * width * height],
        width: width,
        pitch: 3 * width, // we use no padding between rows
        height: height,
        format: PixelFormat::RGB,
    };

    // decompress the JPEG data
    decompressor.decompress_(&jpeg_data, img.as_deref_mut())?;

    // use the raw pixel data
    //println!("{:?}", &image.pixels[0..9]);
    Ok(ImgVecU8{
        data: img.pixels,
        height: height as usize,
        width: width as usize})

}
*/

/// Load an image from disk using the `image` crate
pub fn load_image(img_path: &Path) -> Result<ImgVecU8, Box<dyn Error>> {
    println!(
        "Warning: Cette fonction est deprecated: Moins efficace que load_image_and_resize_with_wand3"
    );
    let img = image::ImageReader::open(img_path)?;
    //let res = 518;  // Patch pour resizer l'image dés le chargement, et faire une interpolation bilinéaire
    //println!("lib_files > load_image(): DynamicImage resizing on loading (interp bilineaire triangle): {}x{}", res, res);
    let img = img.decode()?;
    /*                         .resize_to_fill(
        res as u32,
        res as u32,
        image::imageops::FilterType::Triangle,
    );
    */

    let img = img.to_rgb8();
    let (width, height) = img.dimensions();
    let data = img.into_raw();
    Ok(ImgVecU8 {
        data,
        height: height as usize,
        width: width as usize,
    })
}

/// Load an image from disk using the `wand` crate (ImageMagick) and resize
pub fn load_image_and_resize_with_wand3(
    img_path: &str,
    res: usize,
) -> Result<ImgVecU8, Box<dyn Error>> {
    START.call_once(|| {
        magick_wand_genesis();
    });
    let wand = MagickWand::new();
    wand.read_image(img_path)?;
    let old_height = wand.get_image_height();
    let old_width = wand.get_image_width();
    //println!("debug 0 - load_image_and_resize_with_wand: height={}, width={}", old_height, old_width);

    // Cas habituel ou l'image de départ est plus grande sur les 2 dimensions, que l'image d'arrivée
    let (height, width): (usize, usize) = if old_height < old_width {
        (
            res,
            ((res as f32) * (old_width as f32) / (old_height as f32)).round() as usize,
        )
    } else {
        (
            ((res as f32) * (old_height as f32) / (old_width as f32)).round() as usize,
            res,
        )
    };

    let (crop_height, crop_width): (isize, isize) = if old_height < old_width {
        (0, ((width - res) / 2).try_into().unwrap())
    } else {
        (((height - res) / 2).try_into().unwrap(), 0)
    };
    //println!("debug 1 - geometries:{}, {}, {}, {}", height, width, crop_height, crop_width);

    //wand.fit(res, res);

    // https://legacy.imagemagick.org/Usage/filter/
    //wand.resize_image(width, height, magick_rust::FilterType::Point)?; // nearest pixel interp
    //wand.resize_image(width, height, magick_rust::FilterType::Triangle)?; // bilinear interp
    //wand.resize_image(width, height, magick_rust::FilterType::Catrom)?;   // bicubic interp
    wand.resize_image(width, height, magick_rust::FilterType::Lanczos)?; // Lanczos
    //println!("Warning: Test: Utilise Point/nearest pour l'interpolation du resize dans wand.");

    //let height = wand.get_image_height();
    //let width = wand.get_image_width();
    //println!("debug 1 - load_image_and_resize_with_wand: height={}, width={}", height, width);

    wand.crop_image(res, res, crop_width, crop_height)
        .expect("Error while crop");
    let height = wand.get_image_height();
    let width = wand.get_image_width();
    //println!("debug 2 - load_image_and_resize_with_wand: height={}, width={}", height, width);
    if (height != res) || (width != res) {
        let err_msg = format!(
            "Erreur dans le calcul des dimensions d'image:{},{},{},{},{},{}",
            old_height, old_width, height, width, crop_height, crop_width
        );
        return Err(Box::new(io::Error::other(err_msg)));
    }

    let data = wand
        .write_image_blob("jpeg")
        .expect("Error converting to blob");

    let cursor = Cursor::new(data);
    // Charger l'image à partir des données binaires
    let data = ImageReader::new(cursor)
        .with_guessed_format()
        .expect("Failed to guess image format")
        .decode()
        .expect("Failed to decode image")
        .to_rgb8()
        .into_raw();

    Ok(ImgVecU8 {
        data,
        height,
        width,
    })
}

#[allow(dead_code)]
pub fn save_image(img: &ImgVecU8, img_path: &Path) {
    let mut img_buffer = ImageBuffer::new(img.width as u32, img.height as u32);

    let mut index = 0;
    for y in 0..img.height {
        for x in 0..img.width {
            img_buffer.put_pixel(
                x as u32,
                y as u32,
                Rgb([img.data[index], img.data[index + 1], img.data[index + 2]]),
            );
            index += 3;
        }
    }
    img_buffer.save(&img_path).expect("Failed to save image");

    println!("Image enregistrée: {}", img_path.to_string_lossy());
}

/// List safetensors of a folder
pub fn get_folder_tensors_list(
    folder_path: &Path,
) -> Result<impl Iterator<Item = PathBuf>, Box<dyn Error>> {
    get_folder_files_list(folder_path, &SAFETENSORS_EXTENSIONS)
}

pub fn load_tensor(tensor_path: &Path, device: &Device) -> Result<Tensor, Box<dyn Error>> {
    let h = candle_safetensors_load(tensor_path, device)?;
    Ok(h["tensor"].clone())
}

pub fn save_tensor(tensor: &Tensor, tensor_path: &Path) {
    tensor
        .save_safetensors("tensor", Path::new(&tensor_path))
        .expect("Error");
    println!(
        "Fichier tenseur (dims {:?}) enregistré: {}",
        tensor.dims(),
        tensor_path.to_string_lossy().into_owned()
    );
}

pub fn read_text_file(f_path: &str) -> Vec<String> {
    std::fs::read_to_string(f_path)
        .expect("missing file")
        .split('\n')
        .map(|s| s.to_string())
        .collect()
}

pub fn get_basename(s: &String) -> Result<String, Box<dyn Error>> {
    let path = Path::new(s);

    match path.file_name() {
        Some(file_name) => match file_name.to_str() {
            Some(file_name_str) => Ok(String::from(file_name_str)),
            None => Err("Le nom du fichier n'est pas un UTF-8 valide".into()),
        },
        None => Err("Le chemin n'a pas de nom de fichier".into()),
    }
}

/// Struct to represent pc24 predictions as a 'dataframe'
pub struct PredictionsDataframe {
    pub class_list: Vec<String>,
    pub lines: Vec<(String, Tensor)>,
}

impl PredictionsDataframe {
    pub fn from_csv(s_file_path: &str, device: &Device) -> Result<Self, Box<dyn Error>> {
        let file_path = Path::new(s_file_path);
        let file = File::open(file_path)?;
        let mut rdr = ReaderBuilder::new().delimiter(b';').from_reader(file);

        // Lire les en-têtes de colonnes
        let headers = rdr.headers()?.clone();
        let class_list: Vec<String> = headers.into_iter().skip(1).map(|h| h.to_string()).collect();

        let mut lines = Vec::new();

        // Lire les lignes
        for result in rdr.records() {
            let record = result?;
            let image_name = record.get(0).expect("Missing image_name").to_string();
            let values: Vec<f32> = record
                .iter()
                .skip(1)
                .map(|v| v.parse().expect("Parse error"))
                .collect();
            let shape = values.len();
            let tensor = Tensor::from_vec(values, shape, device)?;
            lines.push((image_name, tensor));
        }

        Ok(PredictionsDataframe { class_list, lines })
    }

    pub fn to_csv(&self, s_file_path: &str) -> Result<(), Box<dyn Error>> {
        println!("Appel à PredictionsDataframe.to_csv()");
        let file_path = Path::new(s_file_path);
        let file = File::create(file_path)?;
        //let mut wtr = Writer::from_writer(file);
        let mut wtr = WriterBuilder::new().delimiter(b';').from_writer(file);

        // Écrire les en-têtes de colonnes
        let mut cols_headers = (self.class_list).clone();
        cols_headers.insert(0, String::from("image_name")); // 1ere colonne = les noms d'image
        wtr.write_record(cols_headers).expect("csv write error");

        // Écrire les lignes
        for (label, tensor) in self.lines.iter() {
            let mut record = vec![label.clone()];
            for value in tensor.to_vec1::<f32>()? {
                record.push(value.to_string());
            }
            wtr.write_record(&record).expect("csv write error");
        }

        wtr.flush()?;
        println!("Fichier écrit: {s_file_path:?}");
        Ok(())
    }
}

/// Struct to represent top classes per plot: "plot -> (proba, classe id, class name)"
#[allow(clippy::type_complexity)]
pub struct PredictionsTopClasses {
    pub lines: Vec<(String, Vec<(f32, String, String)>)>,
}

impl PredictionsTopClasses {
    pub fn from_predictions_dataframe(
        predictions: &PredictionsDataframe,
        dict_class_name: &HashMap<String, String>,
        top_n: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let mut lines = Vec::new();

        for (image_name, tensor) in &predictions.lines {
            // Convertir le tenseur en un vecteur de probabilités
            let probs = tensor.to_vec1::<f32>().unwrap();

            // Créer un vecteur de tuples (probabilité, class_id)
            let mut class_probs: Vec<(f32, &String)> = predictions
                .class_list
                .iter()
                .enumerate()
                .map(|(i, class_id)| (probs[i], class_id))
                .collect();

            // Trier par probabilité décroissante
            class_probs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            // Sélectionner les top_n classes
            let top_classes: Vec<(f32, String, String)> = class_probs
                .into_iter()
                .take(top_n)
                .filter_map(|(prob, class_id)| {
                    dict_class_name
                        .get(class_id)
                        .map(|name| (prob, class_id.clone(), name.clone()))
                })
                .collect();

            lines.push((image_name.clone(), top_classes));
        }

        Ok(PredictionsTopClasses { lines })
    }

    pub fn to_csv(&self, s_file_path: &str) -> Result<(), Box<dyn Error>> {
        let file = File::create(s_file_path)?;
        let mut writer = csv::Writer::from_writer(file);

        for (image_name, classes) in self.lines.iter() {
            let mut record = vec![image_name.clone()];
            for (prob, class_id, class_name) in classes {
                record.push(format!("({}, {}, {})", prob, class_id, class_name));
            }
            writer.write_record(&record)?;
        }

        writer.flush()?;
        println!("Fichier écrit: {s_file_path:?}");
        Ok(())
    }
}

/// Loads a CSV file into a `HashMap`, using two specified columns.
/// Returns: `HashMap<String, String>` or an error
pub fn load_csv_to_hashmap(
    file_path: &str,
    label_key: &str,
    label_value: &str,
    delimiter: Option<u8>,
) -> Result<HashMap<String, String>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let delimiter = delimiter.unwrap_or(b';');

    // Construction du lecteur CSV avec le délimiteur
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .from_reader(file);

    // Lire les en-têtes pour trouver les indices des colonnes
    let headers = rdr.headers()?;
    let key_index = headers
        .iter()
        .position(|h| h == label_key)
        .ok_or("Label key not found")?;
    let value_index = headers
        .iter()
        .position(|h| h == label_value)
        .ok_or("Label value not found")?;

    let mut map = HashMap::new();

    // Itérer sur les enregistrements et remplir le HashMap
    for result in rdr.records() {
        let record = result?;
        let key = record[key_index].to_string();
        let value = record[value_index].to_string();
        map.insert(key, value);
    }

    Ok(map)
}
