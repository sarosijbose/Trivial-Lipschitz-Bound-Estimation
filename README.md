# Trivial-Lipschitz-Bound-Estimation

## Description
Adversarial robustness of a network is defined by how much a certain trained neural network can withstand input perturbations. It was first brought to the notice of the Deep learning community with this [paper](https://arxiv.org/pdf/1312.6199.pdf) and established the Lipschitz constant as a reliable measure for how much a network can withstand such adversarial attacks. Subsequent [work](https://proceedings.neurips.cc/paper/2018/file/d54e99a6c03704e95e6965532dec148b-Paper.pdf) also proved that finding out the Lipschitz constant of a network is NP-Hard. Since then a barrage of methods have come up with the intention of approximating (mostly upper but in some cases lower bound too) the Lipschitz constant for a network. This work provides the code for the naive upper bound Lipschitz constant estimation of any fully connected neural network. The obtained value may be used as a certificate of robustness against adversarial attacks though other far more powerful SOTA theorems exist. It is intended to provide a good but simple entry into this particular domain since this field is known to be extremely theoritical and has a high entry barrier for beginners.

## Setup:-
1. It is recommended to setup a fresh virtual environment first.
```bash
python3 -m venv env
source activate env/bin/activate
```
Then install the required dependencies.

```bash
pip install -r requirements.txt
``` 
2. Given utilities:-

A sample ```random_weights.mat``` file has been given to test the code. Users can plug in other weights (both random or pre-trained) without any difficulties. Here is a simple code snippet showing how to generate random weights for *resembling* a single hidden-layer neural network.

```python
import numpy as np
from scipy.io import savemat
import os

net_dims = [2, 10, 10] 
fname = os.path.join(os.getcwd(), '..../random_weights.mat')
weight_list = [np.random.rand(net_dims[1], net_dims[0]), np.eye(net_dims[1])]
arr = np.empty(2, dtype=object)
arr[:] = weight_list
data = {'weights': arr}
savemat(fname, data)
```
NOTE: For WINDOWS Users, a certain redistributable may be needed otherwise which runtime errors may generate while parsing the network. It has therefore also been provided.

## Running the code:-

```bash
python Lipschitz_trivial_bound.py --weightpath ...utils/random_weights.mat
```
which should generate something like this,

```bash
INFO:root:Size of weight network is 4
INFO:root:Lipschitz bound for layer 1 is 0.963
INFO:root:Lipschitz bound for layer 2 is 4.377
INFO:root:Lipschitz bound for layer 3 is 6.358
INFO:root:Lipschitz bound for layer 4 is 1.615
The trivial upper Lipschitz bound for the network is 43.306
```
## Future Work and Applications:-
The value obtained can be used for feedback control in deep reinforcement learning based systems. Other applications include penalizing GANs for producing stricter discriminator outputs. 
There is also a lot of scope to extend this work. Multi-CPU or GPU implementations would be a good addition since naive estimations scale very poorly for large networks. 
