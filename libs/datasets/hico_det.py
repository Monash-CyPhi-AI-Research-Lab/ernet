import json
import numpy as np
import os
import os.path as osp
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random

import torch
from torch.utils.data import Dataset


class HICODetDataset(Dataset):

    def __init__(self,
                 cfg,
                 data_root,
                 transform=None,
                 istrain=False,
                 ):
        """
        Args:
            data_root: absolute root path for train or val data folder
            transform: train_transform or eval_transform or prediction_transform
        """
        self.num_classes_verb = cfg.DATASET.REL_NUM_CLASSES
        self.data_root = data_root
        self.labels_path = osp.join(osp.abspath(
            self.data_root), 'anno.json')
        self.transform = transform
        self.hoi_annotations = json.load(open(self.labels_path, 'r'))
        self.ids = []
        for i, hico in enumerate(self.hoi_annotations):
            flag_bad = 0
            if len(hico['annotations']) > cfg.TRANSFORMER.NUM_QUERIES:
                flag_bad = 1
                continue
            for hoi in hico['hoi_annotation']:
                if hoi['subject_id'] >= len(hico['annotations']) or hoi[
                     'object_id'] >= len(hico['annotations']):
                    flag_bad = 1
                    break
            if flag_bad == 0:
                self.ids.append(i)
        self.neg_rel_id = 0

        # Number of images per class
        self.cls_num_list = np.zeros(cfg.DATASET.REL_NUM_CLASSES+1) 
        for i in range(len(self.ids)):
            ann_id = self.ids[i]
            hoi_anns = self.hoi_annotations[ann_id]['hoi_annotation']

            hoi_idx = []
            for j in range(len(hoi_anns)):
                hoi_cat = hoi_anns[j]['category_id']
                if isinstance(hoi_cat, list):
                    if len(hoi_cat)>1:
                        for k in range(len(hoi_cat)):
                            hoi_idx.append(hoi_cat[k])
                    else:
                        hoi_idx.append(hoi_cat[0])
                else:
                    hoi_idx.append(hoi_cat)
            hoi_idx = np.unique(np.array(hoi_idx))
            for j in range(len(hoi_idx)):
                self.cls_num_list[hoi_idx[j]]+=1
        self.cls_num_list_path = osp.join(cfg.DATASET.ROOT, 
                                'cls_num_list.npy')
        np.save(self.cls_num_list_path,self.cls_num_list)

    def __len__(self):
        return len(self.ids)

    def multi_dense_to_one_hot(self, labels, num_classes):
        num_labels = labels.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels.ravel()] = 1
        one_hot = np.sum(labels_one_hot, axis=0)[1:]
        in_valid = np.where(one_hot>1)[0]
        one_hot[in_valid] = 1
        return one_hot

    def __getitem__(self, index):
        ann_id = self.ids[index]
        file_name = self.hoi_annotations[ann_id]['file_name']
        img_path = os.path.join(self.data_root, file_name)

        anns = self.hoi_annotations[ann_id]['annotations']
        hoi_anns = self.hoi_annotations[ann_id]['hoi_annotation']
        
        if not osp.exists(img_path):
            logging.error("Cannot found image data: " + img_path)
            raise FileNotFoundError
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        num_object = len(anns)
        num_rels = len(hoi_anns)
        boxes = []
        labels = []
        no_object = False
        if num_object == 0:
            # no gt boxes
            no_object = True
            boxes = np.array([]).reshape(-1, 4)
            labels = np.array([]).reshape(-1,)
        else:
            for k in range(num_object):
                ann = anns[k]
                boxes.append(np.asarray(ann['bbox']))
                if isinstance(ann['category_id'], str):
                    ann['category_id'] =  int(ann['category_id'].replace('\n', ''))
                cls_id = int(ann['category_id'])
                labels.append(cls_id)
            boxes = np.vstack(boxes)
        
        boxes = torch.from_numpy(boxes.reshape(-1, 4).astype(np.float32))
        labels = np.array(labels).reshape(-1,)
        target = dict(
            boxes=boxes,
            labels=labels
        )
        if self.transform is not None:
            img, target = self.transform(
                img, target
            )
        target['labels'] = torch.from_numpy(target['labels']).long()
        boxes = target['boxes']

        hoi_labels = []
        hoi_vecs = []
        hoi_boxes = []
        if num_object == 0:
            hoi_vecs = torch.from_numpy(np.array([]).reshape(-1, 4))
            hoi_boxes = torch.from_numpy(np.array([]).reshape(-1, 4))
            hoi_labels = np.array([]).reshape(-1, self.num_classes_verb)
        else:
            for k in range(num_rels):
                hoi = hoi_anns[k]
                if not isinstance(hoi['category_id'], list):
                    hoi['category_id'] = [hoi['category_id']]
                hoi_label_np = np.array(hoi['category_id'])
                if 'vcoco' in self.data_root:
                    hoi_label_np = hoi_label_np+1
                hoi_labels.append(self.multi_dense_to_one_hot(hoi_label_np,
                                                              self.num_classes_verb+1))
                # hoi vectors
                sub_ct_coord = boxes[hoi['subject_id']][..., :2]     
                obj_ct_coord = boxes[hoi['object_id']][..., :2]
                hoi_vecs.append(torch.cat([sub_ct_coord, obj_ct_coord], dim=-1).reshape(-1, 4))

                # hoi boxes
                sub_wh_coord = boxes[hoi['subject_id']][..., 2:]     
                obj_wh_coord = boxes[hoi['object_id']][..., 2:] 
                sub_box_xyxy = torch.stack([(sub_ct_coord[0] - 0.5 * sub_wh_coord[0]), 
                                        (sub_ct_coord[1] - 0.5 * sub_wh_coord[1]),
                                        (sub_ct_coord[0] + 0.5 * sub_wh_coord[0]), 
                                        (sub_ct_coord[1] + 0.5 * sub_wh_coord[1])], dim=-1)
                obj_box_xyxy = torch.stack([(obj_ct_coord[0] - 0.5 * obj_wh_coord[0]), 
                                        (obj_ct_coord[1] - 0.5 * obj_wh_coord[1]),
                                        (obj_ct_coord[0] + 0.5 * obj_wh_coord[0]), 
                                        (obj_ct_coord[1] + 0.5 * obj_wh_coord[1])], dim=-1)
                hoi_box = torch.Tensor([torch.min(sub_box_xyxy[0], obj_box_xyxy[0]), 
                          torch.min(sub_box_xyxy[1], obj_box_xyxy[1]),
                          torch.max(sub_box_xyxy[2], obj_box_xyxy[2]),
                          torch.max(sub_box_xyxy[3], obj_box_xyxy[3])])
                hoi_box = torch.Tensor([(hoi_box[0] + hoi_box[2]) / 2, 
                                      (hoi_box[1] + hoi_box[3]) / 2,
                                      (hoi_box[2] - hoi_box[0]), 
                                      (hoi_box[3] - hoi_box[1])])    
                hoi_boxes.append(hoi_box)
            hoi_labels = np.array(hoi_labels).reshape(-1, self.num_classes_verb)

        target['rel_labels'] = torch.from_numpy(hoi_labels)
        if len(hoi_vecs) == 0:
            target['rel_vecs'] = torch.from_numpy(np.array([]).reshape(-1, 4)).float()
            target['rel_boxes'] = torch.from_numpy(np.array([]).reshape(-1, 4)).float()
        else:
            target['rel_vecs'] = torch.cat(hoi_vecs).reshape(-1, 4).float()
            target['rel_boxes'] = torch.cat(hoi_boxes).reshape(-1, 4).float()
        
        target['size'] = torch.from_numpy(np.array([h, w]))
        return img, target, file_name
