python -m torch.distributed.run --nproc_per_node 2 tools/train.py --cfg configs/hico.yaml --distributed --dist-url env://
