This file briefly describes how a single plant species prediction model was trained on PlantCLEF2024 training data.

The training data can be found here: https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data/
They can be retrieved via a tar file: https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data/PlantCLEF2024singleplanttrainingdata.tar
or retrieved from the metadata file (named column image_backup_url): https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data/PlantCLEF2024singleplanttrainingdata.csv

In both cases, the image files have been split and organized into 3 subsets: 
- train: for learning, 
- val to validate, control any risk of overfitting and select the best model during training
- test*: to check species identification performance and generalization capability on images of plants alone 

*Warning: although this subset of the data is called "test", it is not the PlantCLEF2024 test set that will be used to compare plant plot identification methods on the basis of high resolution images containing several species. It's a test set dedicated to plants alone.


Based on these 3 train/va/test subsets, several models have been pre-trained, to enable participants who don't have access to high-performance GPU machines to take part in the challenge, focusing on adapting the models to multi-species predictions on high resolution vegetative plot images. 

The first pre-trained model is based on a ViT base patch 14 architecture pre-trained with the SSL (Self-Supervised Learning) Dinov2 method (https://arxiv.org/pdf/2309.16588.pdf), where the backbone has been frozen, and only one classification head has been finetuned on the data.
This current second pre-trained model continue training of the previous pre-trained model but on the entire model, backbone and classification head.

The model was trained with the timm library (version 0.9.16) under torch (version 2.2.1+cu121): 
https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m

The args.yaml file indicates the hyperparameters used.

The summary file shows loss and accuracies progressions during training.

The file class_mapping.txt gives the correspondences between the ids of the model output and the species ids indicated in the fihcier PlantCLEF2024singleplanttrainingdata.csv. For example, model output 2 corresponds to a logit (or probability if using a softmax) of species 1355870 (i.e. the species "Crepis foetida L.").

The weights of the pre-trained model are stored in the file model_best.pth.tar

The model has been trained with the Exponential Moving Average (EMA) option for even better performances.

For your convenience, we're also sharing a basic_usage_pretrained_model.py file as an example of how to use the pre-trained model in inference mode. 



 
