#!/bin/bash
python BigGAN-PyTorch/train.py \
--dataset I64_hdf5 --parallel --shuffle  --num_workers 8 --batch_size 128  \
--num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 32 --D_attn 32 \
--G_nl relu --D_nl relu \
--SN_eps 1e-8 --BN_eps 1e-5 --adam_eps 1e-8 \
--G_ortho 0.0 \
--G_init xavier --D_init xavier \
--G_eval_mode \
--G_ch 32 --D_ch 32 \
--ema --use_ema --ema_start 2000 \
--test_every 5000 --save_every 5000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--which_best FID --num_iters 500000 --num_epochs 1000 \
--embedding inceptionv3 --density_measure gaussian --retention_ratio 100 \
--maximum_focusing_rate 0.5
