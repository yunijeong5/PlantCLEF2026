# File tree from project root

```
PlantCLEF2026
├── README.md
├── data (gitignored)
│   ├── test
│   ├── train_pseudoquadrats
│   └── train_singleplant
├── pretrained_models (gitignored)
│   ├── basic_usage_pretrained_model.py
│   ├── bd2d3830ac3270218ba82fd24e2290becd01317c.jpg
│   ├── class_mapping.txt
│   ├── species_id_to_name.txt
│   ├── vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier
│   └── vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all
└── share
    └── pretrained_model_usage
```

# File tree from data

```
data
├── test
│   ├── images
│   └── PlantCLEF2025_test.csv (available on Kaggle)
├── train_pseudoquadrats
│   ├── LUCAS2006 (subdirectories like: LUCAS*/<site code>/<id1>/<id2>/*.jpg)
│   ├── LUCAS2009 (what id1 and id2 means is unclear)
│   ├── LUCAS2012
│   ├── LUCAS2015
│   ├── LUCAS2018
│   └── pseudoquadrats_without_labels_complementary_training_set_urls.csv (available on Kaggle)
└── train_singleplant
    ├── test (subdirectories like: test/<species_id>/*.jpg)
    ├── train
    ├── val
    ├── PlantCLEF2024_single_plant_training_metadata.csv (available on Kaggle)
    └── class_mapping.txt
```
