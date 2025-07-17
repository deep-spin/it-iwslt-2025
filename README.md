[![arxiv preprint](https://img.shields.io/badge/arXiv-2208.01575-b31b1b.svg)](https://arxiv.org/abs/2506.17019)

Code for the paper: **Instituto de Telecomunicações at IWSLT 2025: Aligning Small-Scale Speech and Language Models for Speech-to-Text Learning**.


> [!NOTE]
> We are working to release an improved training and inference codebase. 
> In this repository, you will only find the model implementation and training code and configs for our IWSLT 2025 submission.

## Project Structure

The following list is an overall description of the main folders and scripts in the repository. We are not releasing scripts to download and prepare datasets locally. Feel free to reach out if you want to replicate our exact setup.

* `bash/`: contains a bash script to schedule a training run using SLURM.
* `config/`: the bash runner requires two yaml configuration files, one to control the distributed training using Hugging Face's accelerate, one for the training parameters. These files can be found here.
* `src/`: contains the training utilities and scripts, as well as the modeling code (folder: `speechlm`). The model is implemented using Hugging Face transformers.

## Point of Contact

For inquiries, feel free to email [Giuseppe Attanasio](mailto:giuseppeattanasio6@gmail.com).