python -m torch.distributed.run --nproc_per_node 2 tools/train.py --cfg configs/vcoco.yaml --distributed --dist-url env://
