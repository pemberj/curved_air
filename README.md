# curved_air

A repository of code developed during my PhD thesis, applicable to several areas of
optical design tasks.

Contains code for optimisation of compound prisms (in particular for development of
atmospheric dispersion correctors [ADCs]); narrowing down the options for the main
parameters of an echelle spectrograph; and for numerical simulation of Fabry-Pérot
etalons.


## Installation

I recommend using [`uv`](https://astral.sh/uv) to manage Python packages, dependencies
and virtual environments. To run code in this repository with `uv`:

`git clone https://github.com/pemberj/curved_air`

And simply:

`uv run ./example_scripts/spectrographs/MARVEL.py`

`uv` will pull and install all the relevant dependencies into a virtual environment and
run the code.
