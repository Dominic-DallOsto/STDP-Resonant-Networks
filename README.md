# STDP-Resonant-Networks

Simulating spiking networks with STDP and homeostatic learning rules, as described in [Aceituno, 2020](http://arxiv.org/abs/2006.08537).

## Installation

A custom fork of the [Bindsnet](https://github.com/BindsNET/bindsnet) package is used for spiking network simulations. Install it like so (preferably in a separate conda environment).

```bash
git submodule update
pip install -e ./bindsnet
```

Then run any of the notebook examples
