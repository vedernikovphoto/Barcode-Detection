# Barcode Detection Project
This project focuses on detecting barcodes in images. The implementation uses PyTorch Lightning for model training pipelines, ClearML for experiment tracking, and DVC for model versioning and data management.

This is **Part 1** of the barcode processing pipeline. The detected barcodes can be further processed using the [Barcode OCR Project](https://github.com/vedernikovphoto/Barcode-OCR) for text extraction.


## Installation
1. **Clone the repository:**
   ```sh
   git clone https://github.com/vedernikovphoto/Barcode-Detection.git
   cd Barcode-Detection
   ```

2. **Ensure you have Python 3.10 installed and set up your environment. Then, install the required packages:**
    ```sh
    make install
    ```


## Download Dataset
To download and extract the dataset:
```sh
make download
```
This command will download the dataset and place it in the `data` directory located in the root of the project.


## Training the Model
To start training the model:
```sh
make train
```
This command will initiate the training process using the configuration specified in `config/config.yaml`.


## Experiment Tracking
We use ClearML for tracking experiments. You can view the metrics and configurations of the experiments on ClearML. Access the experiment details [here](https://app.clear.ml/projects/95147d68e3b44dfcaaac1ff7caab5240/experiments/67bc39c164f849e0aacfe80807627eb1/output/execution). 

Ensure that your ClearML credentials are properly configured before running the experiments.


## Model Checkpoints
Model checkpoints are saved periodically during training in the `experiments` directory. These checkpoints can be used to resume training or for inference. The checkpoint files are named based on the epoch and validation F1 score.


## Resuming Training
If the training process is interrupted, you can resume training from the last saved checkpoint. Ensure that the checkpoint file is located in the `experiments` directory, then restart the training using the appropriate command.


## Model Versioning

We use DVC for model versioning. The model weights are stored in a private remote storage configured through DVC. If you have access to this storage, follow the steps below to retrieve the weights:

1. **Configure the DVC remote storage:**
   Ensure your DVC remote storage is correctly configured. Run the following commands:

   ```bash
   dvc remote add --default <remote_name> <remote_url>
   dvc remote modify <remote_name> user <username>
   dvc remote modify <remote_name> keyfile <path_to_private_key>
    ```

2. **Pull the latest weights:**
    Run the following command to pull the latest model weights from the remote storage:
    ```bash
    dvc pull experiments
    ```

3. **Verify the files:** 
    After pulling the files, verify them using:
    ```bash
    ls experiments/experiment1
    ```
    You should see the model weight files such as `model.pt`.

This process ensures that you have the necessary files in your local setup for inference or further training. For inference, place the downloaded file into the `model_weights` directory.

If you do not have your own DVC remote storage, you can request the model weights by sending an email to [alexander.vedernikov.edu@gmail.com](mailto:alexander.vedernikov.edu@gmail.com). Once you receive the weights, place them in the `model_weights` directory.


## Running Inference

To run inference on new images, place the images in the `inference_images` directory. Then, execute the following command from the root of the project:

```sh
make inference
```


This will:
- Use the model weights from `model_weights` folder.
- Perform inference on images in the `inference_images` folder.
- Save results (images with predictions) in the `output_images/results` folder.
- Save prediction details (text files with detected objects and confidence scores) alongside the output images.



## Linting

The project uses `wemake-python-styleguide` for linting, configured in `src/setup.cfg`. To lint, navigate to `src` and run:

```sh
cd src
flake8 .
```
