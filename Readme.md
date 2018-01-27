# Babysitting an Inception-Resnet-V2 food classifier
## Achieving state of the art accuracy in food classification using an Inception-Resnet-V2 network. Built to train and evaluate on multiple GPUs.


### Dataset Preparation
The primary dataset used in this paper to perform classification is the Food-101 dataset[1].
The same procedure can be applied to the UECFood-256[2] and UECFood-100 datasets[3]. 

* Begin by downloading and extracting the Food-101 dataset:
    ```bash
    cd Dataset
    curl http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
    tar -xzf food-101.tar.gz
    cd ..
    ```

* Convert the dataset into TFrecord files for easily feeding into data pipeline (takes ~5mins on 1060 TI gpu):
    ```bash
    cd Code
    python3 convert_dataset.py --tfrecord_filename=foods --dataset_dir="../Dataset/food-101/images"
    cd ..
    ```

### Train the model

* Download a pre-trained model, or rename the included `pretrained-inception-resenet-v2` model to fit your needs.

* Run the model training process. This code will evaluate the model every epoch for 50 steps.
    ```bash
    python model.py --model=pretrained-inception-resnet-v2 --dataset=../Dataset/FOOD101/images
    ```



### References
 * [1] - https://www.vision.ee.ethz.ch/datasets_extra/food-101/
 * [2] - http://foodcam.mobi/dataset256.html
 * [3] - http://foodcam.mobi/dataset100.html
 