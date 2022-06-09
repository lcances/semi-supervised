# Semi Supervised Learning - Deep Co-Training

Application of Deep Co-Training for audio tagging on multiple audio dataset.

# Requirements
```bash
git clone https://github.com/leocances/semi-supervised.git

cd semi-supervised
conda env create -f environement.ym
conda activate ssl

pip install -e .

```
<!--
## Manually
```bash
conda create -n dct python=3 pip
conda activate dct

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install numpy
conda install pandas
conda install scikit-learn
conda install scikit-image
conda install tqdm
conda install h5py
conda install pillow
conda install librosa -c conda-forge

pip install hydra-core
pip install advertorch
pip install torchsummary
pip install tensorboard

cd Deep-Co-Training
pip install -e .
```
## Fix missing package
- It is very likely that the `ubs8k` will be missing. It a code to manage the UrbanSound8K dataset I wrote almost two years ago before I start using `torchaudio`.
- `pytorch_metrics` is a basic package I wrote to handle many of the metrics I used during my experiments.
- `augmentation_utils` is a package I wrote to test and apply many different augmentation during my experiments.
```bash
pip install --upgrade git+https://github.com/leocances/UrbanSound8K.git@
pip install --upgrade git+https://github.com/leocances/pytorch_metrics.git@v2
pip install --upgrade git+https://github.com/leocances/augmentation_utils.git
```
I am planning on release a much cleaner implementation that follow the torchaudio rules.
-->

## Reproduce full supervised learning for UrbanSound8k dataset
```bash
conda activate dct
python standalone/full_supervised/full_supervised.py --from-config DCT/util/config/ubs8k/100_supervised.yml
```

# Train the systems
The directory `standalone/` contains the scrpits to execute the different semi-supervised methods and the usual supervised approach. Each approach has it own working directory which contain the python scripts.

The handling of running arguments is done using [hydra](hydra.cc) and the configuration files can be found in the directory `config/`. There is one configuration file for each dataset and methods.

## Train for speechcommand
```bash
conda activate ssl
cd semi-supervised/standalone/supervised

python supervised.py -cn ../../config/supervised/speechcommand.yaml
```

You can override the parameters from the configuration file by doing the following, allowing you to change the model to use, training parameters or some augmentation parameters. please read the configuration file for more detail.

## Train for speechcommand using ResNet50
```bash
python supervised.py -cn ../../config/supervised/speechcommand.yaml model.model=resnet50
```

# Basic API [WIP]
I am currently trying to make a main script from which it will possible to train and use the models easily.
This documentation is more for my personnal use and is not exaustif yet. It is better to use directly the proper training script with the conrresponding configuration file.

## commands
- **train**
    - **--model**: The model architecture to use. See [Available models](#available-models)
    - **--dataset**: The dataset you want to use for the training. See [Install datasets](#install-datasets)
    - **--method**: The learning method you want to use for the training. See [Available methods](#available-methods)
    - \[hydra-override\]
    - **Exemple**
    ```bash
    python run_ssl train --dataset speechcommand --model wideresnet28_2 --method mean-teacher [hydra-override-args ...]
    ```

- **inference**
    - **--model**: The model architecture to use. See [Available models](#available-models)
    - **--dataset**: The dataset used to train the model. See [Install datasets](#install-datasets)
    - **--method**: The learning method used for the training. See [Available methods](#available-methods)
    - **-w | --weights**: The path to the weight of the model. If left empty will use the latest file available
    - **-f | --file**: The path to the file that will be fed to the model
    - **-o | --output**: The output expected from \{logits | softmax | sigmoid | pred\}
    - **--cuda**: Use the GPU if this flag is added
    - **Exemple**
    ```bash
    python run_ssl inference \
        --dataset ComParE2021-PRS \
        -o softmax \
        -f ../datasets/ComParE2021-PRS/dist/devel_00001.wav \
        -w ../model_save/ComParE2021-PRS/supervised/wideresnet28_2/wideresnet28_2__0.003-lr_1.0-sr_50000-e_32-bs_1234-seed.best
    ```

- **cross-validation**
    - WIP    


## Available models
WIP

## Install datasets
WIP

## Available methods
WIP
