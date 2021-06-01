#!/bin/bash
python BigGAN-PyTorch/make_hdf5.py --dataset I256 --batch_size 256 --data_root data
python BigGAN-PyTorch/calculate_inception_moments.py --dataset I256_hdf5 --data_root data
python BigGAN-PyTorch/calculate_image_manifold.py --dataset I256_hdf5 --data_root data