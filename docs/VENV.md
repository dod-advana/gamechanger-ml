# Using a Conda Virtual Environment

## Download and Set Up

- See [here](https://www.anaconda.com/products/distribution).

## Create a Virtual Environment

```
conda create -n <env name> python=<version>
```

- Replace \<env name\> with the name you want to give this environment.
- Replace \<version\> with the Python version you want for this environment. The Python version for this repository can be found in [setup.py](../setup.py).

## Activate a Virtual Environment

```
conda activate <env name>
```

- Replace \<env name\> with the name of the environment to activate.

## Deactivate a Virtual Environment

```
conda deactivate
```
