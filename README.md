# MSOS - Adaptive Diffusion Sampling: A Meta-Learning Framework for Automated
Schedule Optimization

This repository provides a streamlined implementation of Stable Diffusion v2.1, with a focus on accelerated inference through the use of the `stages_step_optim.py` script. This allows for faster generation of high-quality images from text prompts.

## Requirements

To get started, you'll need to install the necessary Python packages and download the Stable Diffusion v2.1 model checkpoints.

### 1\. Installation

It is highly recommended to use a virtual environment.

```bash
git clone https://github.com/your-username/msos.git
cd msos

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**Step 1: Install PyTorch**

First, you need to install PyTorch separately, as the specific version depends on your hardware (especially if you have an NVIDIA GPU with CUDA). The `requirements.txt` file does not include PyTorch for this reason.

  * **Visit the official [PyTorch website](https://pytorch.org/get-started/locally/)** to get the correct installation command for your system (select your OS, package manager, and CUDA version or CPU).

  * For example, a common command for systems with CUDA 12.1 is:

    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

**Step 2: Install Dependencies from `requirements.txt`**

Once PyTorch is installed, install all other required packages using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 2\. Download Model Checkpoints

This implementation uses the Stable Diffusion v2.1 model. You will need to download the model checkpoints from the official Stability AI repository on Hugging Face:

  * **Stable Diffusion 2.1-v (768x768):** [Download here](https://huggingface.co/stabilityai/stable-diffusion-2-1)
  * **Stable Diffusion 2.1-base (512x512):** [Download here](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)

Place the downloaded `.ckpt` files in a `checkpoints` directory in the root of the repository, or note the path for the configuration in the next step.

## Usage

Image generation is performed using the **`test_unipc_optim.py`** script. Unlike typical command-line scripts, you must **edit the script directly** to configure the paths and settings.

### 1\. Configure the Script

Open `test_unipc_optim.py` in a text editor and modify the following configuration variables at the top of the file:

```python
# --- Configuration ---
# Folder containing your .txt prompt files
prompts_folder = "your prompt"

# Folder where generated images will be saved
output_folder = "your output path" 

# Path to the model's configuration file
config_path = "configs/stable-diffusion/v2-inference.yaml"

# Path to your downloaded Stable Diffusion checkpoint
ckpt_path = "./models/v2-1_512-ema-pruned.ckpt"

# Number of inference steps (NFE)
n_steps = 4
```

### 2\. Prepare Prompts

Create a folder (e.g., `prompts`) and place your text prompts inside it, with each prompt in a separate `.txt` file. The script will read all `.txt` files from the `prompts_folder` you configured.

### 3\. Run Inference

After configuring the paths in the script, run it from your terminal:

```bash
python test_unipc_optim.py
```

The script will load the model, generate an optimized timestep schedule (or load a cached one), and process each prompt file to generate an image. The resulting images will be saved in the `output_folder` you specified.

## Acceleration with `stages_step_optim.py`

The core of the accelerated inference is the `stages_step_optim.py` script. This script implements an optimized sampling schedule that significantly reduces the number of steps required for image generation without a substantial loss in quality. The `test_unipc_optim.py` script automatically utilizes this to generate or load an optimal schedule for the UniPC sampler.

## License

This project is licensed under the MIT License. The Stable Diffusion model weights are licensed under the [CreativeML Open RAIL++-M License](https://www.google.com/search?q=https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/LICENSE-MODEL).

## Acknowledgements

This project is built upon the incredible work of the following teams and individuals:

  * **CompVis** and **RunwayML** for the original Stable Diffusion model.
  * **Stability AI** for training and releasing the Stable Diffusion 2.1 model.
  * The authors of the UniPC sampler for their work on efficient diffusion model sampling.