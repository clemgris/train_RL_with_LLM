
Create conda environment with jax and torch
```console
conda create -c conda-forge -n tRLwLLM pytorch jax
```
If this does not install the cuda-enabled jax and torch try
```console
conda create -n conda-forge -n tRLwLLM pytorch=*=cuda* jax jaxlib=*=cuda*
```
However, this is not necessary if you are using a recent conda install that defaults to use the conda-libmamba-solver, see https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community .