# ERNet: An Efficient and Reliable Human-Object Interaction Detection Network

<img src="figures/ERNet_Architecture.jpg"  width="800"/>

Abtract: Human-Object Interaction (HOI) detection recognizes how persons interact with objects, which is advantageous in autonomous systems such as self-driving vehicles and collaborative robots. However, current HOI detectors are often plagued by model inefficiency and unreliability when making a prediction, which consequently limits its potential for real-world scenarios. In this paper, we address these challenges by proposing ERNet, an end-to-end trainable convolutional-transformer network for HOI detection. The proposed model employs an efficient multi-scale deformable attention to effectively capture vital HOI features. We also put forward a novel detection attention module to adaptively generate semantically rich instance and interaction tokens. These tokens undergo pre-emptive detections to produce initial region and vector proposals that also serve as queries which enhances the feature refinement process in the transformer decoders. Several impactful enhancements are also applied to improve the HOI representation learning. Additionally, we utilize a predictive uncertainty estimation framework in the instance and interaction classification heads to quantify the uncertainty behind each prediction. By doing so, we can accurately and reliably predict HOIs even under challenging scenarios. Experiment results on the HICO-Det, V-COCO, and HOI-A datasets demonstrate that the proposed model achieves state-of-the-art performance in detection accuracy and training efficiency.

## Installation
Environment
- python >= 3.6

Install the dependencies.
```shell
 pip install -r requirements.txt
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
 - We transformed the original annotation files of HICO-DET to a *.json format, like data/hico/images/train_anno.json and ata/hico/images/test_hico.json.
 - test_hico.json, trainval_hico.json and rel_np.npy are used in the evaluation on HICO-DET. We provided these three files in our data/hico directory.
 - data/hico/train_anno.json and data/hico/images/train/anno.json are the same file.
   `cp data/hico/train_anno.json data/hico/images/train/anno.json`
 - data/hico/test_hico.json and data/hico/images/test/anno.json are the same file.
   `cp data/hico/test_hico.json data/hico/images/test/anno.json`

## Train
To train our model on HICO-DET:
```shell
python -m torch.distributed.launch --nproc_per_node 4 --nnodes=1 tools/train.py --cfg configs/hico.yaml
```

## Evaluation
To evaluate our model on HICO-DET:
```shell
CUDA_VISIBLE_DEVICES=0 python3 tools/eval.py --cfg configs/hico.yaml MODEL.RESUME_PATH [checkpoint_path]
```
- The checkpoint is saved on HICO-DET with torch==1.9.0.
- Currently support evaluation on single GPU.

## Citation 
This work is currently review and the complete code will be released soon.
