# ERNet

## Install Required Dependencies
### Requirements
  - Linux, CUDA>=11.7, GCC>=11.0

  - PyTorch>= 2.0, 
  ```
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
  ```

  - Install other requirements 
  ```
    pip install -r requirements.txt
  ```

  - Compile MSDA CUDA operators
  ```
    cd ./models/ops
    sh ./make.sh
    # unit test (should see all checking is True)
    python test.py
  ```

## Data preparation
- We first download the [ HICO-DET ](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk " HICO-DET ") dataset.
- The data should be prepared in the following structure:
```
data/hico
   |———  images
   |        └——————train
   |        |        └——————anno.json
   |        |        └——————XXX1.jpg
   |        |        └——————XXX2.jpg
   |        └——————test
   |                 └——————anno.json
   |                 └——————XXX1.jpg
   |                 └——————XXX2.jpg
   └——— test_hico.json
   └——— trainval_hico.json
   └——— rel_np.npy
```
Noted:
 - We transformed the original annotation files of HICO-DET to a *.json format, like data/hico/images/train_anno.json and data/hico/images/test_hico.json.
 - test_hico.json, trainval_hico.json and rel_np.npy are used in the evaluation on HICO-DET. We provided these three files in our data/hico directory.
 - data/hico/train_anno.json and data/hico/images/train/anno.json are the same file.
   `cp data/hico/train_anno.json data/hico/images/train/anno.json`
 - data/hico/test_hico.json and data/hico/images/test/anno.json are the same file.
   `cp data/hico/test_hico.json data/hico/images/test/anno.json`

## Train
To train our model on HICO-DET with 4 GPUs on a single node:
```shell
python3 -m torch.distributed.run --nproc_per_node 4 tools/train.py --cfg configs/hico.yaml --distributed --dist-url env://
```

## Evaluation
To evaluate our model on HICO-DET:
```shell
python3 tools/eval.py --cfg configs/hico.yaml MODEL.RESUME_PATH [checkpoint_path]
```
- Currently support evaluation on single GPU.
- Checkpoint coming soon.

## HOIA
- First download the [ HOIA ](https://drive.google.com/drive/folders/15xrIt-biSmE9hEJ2W6lWlUmdDmhatjKt " HOIA ") dataset. We also provide our transformed annotations in data/hoia. 
- The data preparation and training is following our data preparation and training process for HICO-DET. You need to modify the config file to hoia.yaml.


