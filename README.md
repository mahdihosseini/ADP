# ADP

Patch-level Annotated Database of Digital Pathology images for Histological Tissue Type Classification, as presented by Hosseini *et al.*'s CVPR 2019 paper "Atlas of Digital Pathology: A Generalized Hierarchical Histological Tissue Type-Annotated Database for Deep Learning".

## Differences from CVPR Code
* HTTs with no training examples (i.e. N.G.A, N.G.O, N.G.E, N.G.R, N.G.T) are removed, to prevent infinite class weights
* (Optional) Log-inv-freq used instead of Inv-freq as class weights

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. You will download a copy of the ADP database and train/test the three CNN architectures mentioned in the paper with any of the five taxonomic configurations.

### Prerequisites

Mandatory
* `python` (checked on 3.5)
* `pandas` (checked on 0.23.4)
* `keras` (checked on 2.2.4)
* `numpy` (checked on 1.16.2)
* `tensorflow` (checked on 1.13.1)

Optional
* `matplotlib` (checked on 3.0.2)

### Installing

```
cd folder/to/clone-into
git clone https://github.com/mahdihosseini/ADP.git
```

## Quick Start

First, download the separate ADP database to a local directory (TODO: make this scriptable).

Then, open `demo_01_train.py`, then edit the value of `DATASET_DIR` to the location of that local directory.

Edit the `settings.csv` file as necessary (TODO: include notes on what is permitted for each field).

Next, run the demo script to train
```
cd folder/to/clone-into
python demo_01_train.py
```

(TODO: instructions on evaluating a pre-trained network)

## License

This project is licensed under the GNU General Public License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* @[geifmany](https://github.com/geifmany/cifar-vgg) for his implementation of VGG16 for CIFAR, which we adapted for our VGG16 implementation