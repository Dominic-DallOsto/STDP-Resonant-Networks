# STDP-Resonant-Networks

Simulating spiking networks with STDP and homeostatic learning rules, as described in [Aceituno, 2020](http://arxiv.org/abs/2006.08537).

## Installation

A custom fork of the [Bindsnet](https://github.com/BindsNET/bindsnet) package is used for spiking network simulations. Install it like so (preferably in a separate conda environment).

```bash
git submodule update
pip install -e ./bindsnet
```

Then run any of the notebook examples

Some of the simulations include:

* [Testing the sinusoidal case](./Sine%20wave%20simulation.ipynb)
* [Testing the Gaussian case](./Gaussian%20bump%20simulation.ipynb)
* [Checking the theoretical predictions for the sinusoid](./Steady%20state%20weights%20vs%20theoretical%20-%20asymmetrical%20kernel%20sinusoidal.ipynb)
* [Checking the theoretical predictions for the Gaussian](./Steady%20state%20weights%20vs%20theoretical%20-%20asymmetrical%20kernel.ipynb)
* [Checking the learning of different weight strengths for the Gaussian](./Gaussian%20bump%20simulation%20output%20weights.ipynb)
* [Checking the weights needed for self-sustaining Gaussian activity](./Gaussian%20bump%20stable%20network.ipynb)
