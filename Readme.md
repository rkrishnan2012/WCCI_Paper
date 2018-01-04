# Food-ception
## A convolutional network with an Inception V4 backbone architecture used to classify food images.


### Dataset Preparation
The primary dataset used in this paper to perform classification is the Food-101 dataset[1].

* Begin by downloading and extracting the Food-101 dataset:
    ```
    cd Dataset
    curl http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
    tar -xzf food-101.tar.gz
    cd ..
    ```

* Convert the dataset into TFrecord files for easily feeding into data pipeline (takes ~5mins on 1060 TI gpu):
    ```
    cd Code
    python3 convert_dataset.py --tfrecord_filename=foods --dataset_dir="../Dataset/food-101/images"
    cd ..
    ```

### Train the model

* Download the Inception V4 checkpoint file, extract it to Code folder.
    ```
    cd Code
    wget "http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz"
    tar -xzf "inception_v4_2016_09_09.tar.gz" -C .
    ```

* Run the model training process.
    ```
    python model.py
    ```



### References
 * https://www.vision.ee.ethz.ch/datasets_extra/food-101/
 