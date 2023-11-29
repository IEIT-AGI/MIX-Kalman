# Get started

## Prerequisites

- Linux
- Python 3.8+=
- PyTorch 1.7+=
- CUDA 10.1+
- numpy 1.19.4+=
- transformers 3.4.0+=

There are still some dependent packages not listed,  which wil all be installed in the step 4.

## Installation

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n MIX-Kalman python=3.8 -y
    conda activate MIX-Kalman
    ```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

    ```shell
    conda install pytorch torchvision -c pytorch
    ```

    Note: Make sure that your compilation CUDA version and runtime CUDA version match.
    You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

    `E.g.` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install
    PyTorch 1.6+, you need to install the prebuilt PyTorch with CUDA 10.1.

    ```shell
    conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
    ```

3. Clone the MIX-Kalman repository.

    ```shell
    git clone https://github.com/multimodal-framework/MIX-Kalman.git
    cd MIX-Kalman
   ```

4. Install build requirements and then install MIX-Kalman.

    ```shell
    pip install -r requirements.txt
    pip install -v -e .  # or "python setup.py develop"
    ```

Note:

a. Following the above instructions, MIX-Kalman is installed on `dev` mode , any local modifications made to the code will take effect without the need to reinstall it.

### A from-scratch setup script

Assuming that you already have CUDA 10.1 installed, here is a full script for setting up MIX-Kalman with conda.

```shell
conda create -n MIX-Kalman python=3.8 -y
conda activate MIX-Kalman

conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -y

# install MIX-Kalman
git clone https://github.com/multimodal-framework/MIX-Kalman.git
cd MIX-Kalman
pip install -r requirements.txt
pip install -v -e .
```

### Developing with multiple MIX-Kalman versions

The train and test scripts already modify the `PYTHONPATH` to ensure the script use the MIX-Kalman in the current directory.

To use the default MIX-Kalman installed in the environment rather than that you are working with, you can remove the following line in those scripts

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

## Verification

To verify whether MIX-Kalman and the required environment are installed correctly, we can run sample Python code to initialize a train job (use [LXMERT](https://github.com/IEIT-AGI/MIX-Kalman/tree/master/configs/lxmert/LXMERT.md) as an example):

```shell
cd MIX-Kalman
export PYTHONPATH="your/path/MIX-Kalman" # set Python search path as your MIX-Kalman directory path
python tools/run.py --config-file configs/lxmert/lxmert_vqa.py
```

The above code is supposed to run successfully upon you finish the installation.
