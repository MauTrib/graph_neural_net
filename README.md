# Graph neural networks for the Travelling Salesman Problem

## Overview
### Project structure

```bash
.
├── loaders
|   └── dataset selector
|   └── siamese_loader.py # loading pairs
|   └── tsp_data_generator #Create and Load the graphs and their solutions
├── models
|   └── architecture selector
|   └── layers.py # equivariant block
|   └── base_model.py # powerful GNN Graph -> Graph
|   └── siamese_net.py # GNN to match graphs
├── toolbox
|   └── optimizer and losses selectors
|   └── logger.py  # keeping track of most results during training
|   └── metrics.py # computing scores
|   └── losses.py  # computing losses
|   └── optimizer.py # optimizers
|   └── decoding.py #Transforming edge probability in tours
|   └── utility.py
|   └── maskedtensor.py # Tensor-like class to handle batches of graphs of different sizes
├── commander_tsp.py # main file from the project serving for calling all necessary functions for training and testing
├── trainer.py # pipelines for training and validation
```


## Dependencies
Dependencies are listed in `requirements.txt`. To install, run
```
pip install -r requirements.txt
```
To create datasets, it is required to install [pyconcorde](https://github.com/jvkersch/pyconcorde).
## Training 
Run the main file ```commander_tsp.py```
```
python commander_tsp.py
```
To change options, use [Sacred](https://github.com/IDSIA/sacred) command-line interface and see ```default_tsp.yaml``` for the configuration structure. For instance,
```
python commander_tsp.py with cpu=No data.generative_model=Square01 train.epoch=10 
```
You can also copy ```default_tsp.yaml``` and modify the configuration parameters there. Loading the configuration in ```other.yaml``` (or ```other.json```) can be done with
```
python commander_tsp.py with other.yaml
```
See [Sacred documentation](http://sacred.readthedocs.org/) for an exhaustive reference. 

To save logs to [Neptune](https://neptune.ai/), you need to provide your own API key via the dedicated environment variable.

The model is regularly saved in the folder `runs`.
