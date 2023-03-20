import json
import numpy as np
import os
import matplotlib.pyplot as plt     

class vcoco():
    def __init__(self, annotation_file):
        self.annotations = json.load(open(annotation_file, 'r'))
        self.overlap_iou = 0.5
        self.verb_name_dict = {1: 'hold_obj', 2: 'stand', 3: 'sit_instr', 4: 'ride_instr', 5: 'walk', 
                               6: 'look_obj', 7: 'hit_instr',  8: 'hit_obj', 9: 'eat_obj',
                               10: 'eat_instr', 11: 'jump_instr', 12: 'lay_instr', 13: 'talk_on_phone_instr', 14: 'carry_obj',
                               15: 'throw_obj', 16: 'catch_obj', 17: 'cut_instr', 18: 'cut_obj', 19: 'run', 
                               20: 'work_on_computer_instr', 21: 'ski_instr', 22: 'surf_instr', 23: 'skateboard_instr', 24: 'smile', 
                               25: 'drink_instr', 26: 'kick_obj', 27: 'point_instr', 28: 'read_obj',  29: 'snowboard_instr'}
        
        self.verb_name_dict_valid = (np.array([1,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,25,26,28,29]) - 1).tolist()                     
        self.fp = {}
        self.tp = {}
        self.score = {}
        self.sum_gt = {}
        for i in list(self.verb_name_dict.keys()):
            self.fp[i] = []
            self.tp[i] = []
            self.score[i] = []
            self.sum_gt[i] = 0
        self.file_name = []
        for gt_i in self.annotations:
            self.file_name.append(gt_i['file_name'])
            gt_hoi = gt_i['hoi_annotation']
            for gt_hoi_i in gt_hoi:
                if isinstance(gt_hoi_i['category_id'], str):
                    gt_hoi_i['category_id'] = int(gt_hoi_i['category_id'].replace('\n', ''))+1
                else:
                    gt_hoi_i['category_id'] = gt_hoi_i['category_id'] + 1
                if gt_hoi_i['category_id'] in list(self.verb_name_dict.keys()):
                    self.sum_gt[gt_hoi_i['category_id']] += 1
        self.num_class = len(list(self.verb_name_dict.keys()))

    def evalution(self, predict_annot):
        for pred_i in predict_annot:
            if pred_i['file_name'] not in self.file_name:
                continue
            gt_i = self.annotations[self.file_name.index(pred_i['file_name'])]
            gt_bbox = gt_i['annotations']
            pred_bbox = pred_i['predictions']
            pred_hoi = pred_i['hoi_prediction']
            gt_hoi = gt_i['hoi_annotation']
            bbox_pairs = self.compute_iou_mat(gt_bbox, pred_bbox)
            self.compute_fptp(pred_hoi, gt_hoi, bbox_pairs)
        map = self.compute_map()
        return map

    def compute_map(self):
        ap = np.zeros(self.num_class)
        max_recall = np.zeros(self.num_class)

        total_rec = []
        total_prec = []

        for i in list(self.verb_name_dict.keys()):
            sum_gt = self.sum_gt[i]
            if sum_gt == 0:
                continue
            tp = np.asarray((self.tp[i]).copy())
            fp = np.asarray((self.fp[i]).copy())
            res_num = len(tp)
            if res_num == 0:
                continue
            score = np.asarray(self.score[i].copy())
            sort_inds = np.argsort(-score)
            fp = fp[sort_inds]
            tp = tp[sort_inds]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / sum_gt
            prec = tp / (fp + tp)
            ap[i - 1] = self.voc_ap(rec,prec)
            max_recall[i-1] = np.max(rec)
            total_rec.append(rec)
            total_prec.append(prec)

        self.pr_curve(total_rec,total_prec)
        mAP = np.mean(ap[self.verb_name_dict_valid])
        m_rec = np.mean(max_recall[self.verb_name_dict_valid])
        print('--------------------')
        print('mAP: {}   max recall: {}'.format(mAP, m_rec))
        print('--------------------')
        return mAP

    def voc_ap(self, rec, prec):
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def pr_curve(self, rec, prec):
        rec_interp = np.linspace(0.,1.,1000)
        total_pre = np.zeros(1000)
        num_class = len(self.verb_name_dict_valid)

        for i in range(num_class):
            mrec = np.concatenate((rec[i], [1.]))
            mpre = np.concatenate((prec[i], [0.]))

            pre_interp = np.interp(rec_interp,mrec,mpre)

            total_pre+=pre_interp

            # plt.plot(rec_interp,pre_interp)      
            # plt.xlabel('Recall')
            # plt.ylabel('Precision')
            # plt.title('Precision-Recall Curve of {}'.format(self.verb_name_dict[self.verb_name_dict_valid[i]+1]))
            # plt.grid()
            # plt.show()

        plt.plot(rec_interp,total_pre/num_class)      
        plt.xlabel('Recall', fontsize=13)
        plt.ylabel('Precision', fontsize=13)
        plt.title('Average Precision-Recall Curve of V-COCO Interactions',fontweight='bold')
        plt.grid(which='both')
        plt.savefig('VCOCO_PR.png', dpi=300)
        plt.show()

    def compute_fptp(self, pred_hoi, gt_hoi, match_pairs):
        pos_pred_ids = match_pairs.keys()
        vis_tag = np.zeros(len(gt_hoi))
        pred_hoi.sort(key=lambda k: (k.get('score', 0)), reverse=True)

        if len(pred_hoi) != 0:
            for i, pred_hoi_i in enumerate(pred_hoi):
                is_match = 0
                if isinstance(pred_hoi_i['category_id'], str):
                    pred_hoi_i['category_id'] = int(pred_hoi_i['category_id'].replace('\n', ''))
                
                if len(match_pairs) != 0 and pred_hoi_i['subject_id'] in pos_pred_ids and pred_hoi_i['object_id'] in pos_pred_ids:
                    pred_sub_ids = match_pairs[pred_hoi_i['subject_id']]
                    pred_obj_ids = match_pairs[pred_hoi_i['object_id']]
                    pred_category_id = pred_hoi_i['category_id']
                    for gt_id in np.nonzero(1 - vis_tag)[0]:
                        gt_hoi_i = gt_hoi[gt_id]        

                        if (gt_hoi_i['subject_id'] in pred_sub_ids) and (gt_hoi_i['object_id'] in pred_obj_ids) and (pred_category_id == gt_hoi_i['category_id']):
                            is_match = 1
                            vis_tag[gt_id] = 1
                            continue

                if pred_hoi_i['category_id'] not in list(self.fp.keys()):
                    continue
                if is_match == 1:
                    self.fp[pred_hoi_i['category_id']].append(0)
                    self.tp[pred_hoi_i['category_id']].append(1)

                else:
                    self.fp[pred_hoi_i['category_id']].append(1)
                    self.tp[pred_hoi_i['category_id']].append(0)
                self.score[pred_hoi_i['category_id']].append(pred_hoi_i['score'])

    def compute_iou_mat(self, bbox_list1, bbox_list2):
        iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
        if len(bbox_list1) == 0 or len(bbox_list2) == 0:
            return {}
        for i, bbox1 in enumerate(bbox_list1):
            for j, bbox2 in enumerate(bbox_list2):
                iou_i = self.compute_IOU(bbox1, bbox2)
                iou_mat[i, j] = iou_i
        iou_mat[iou_mat>= self.overlap_iou] = 1
        iou_mat[iou_mat< self.overlap_iou] = 0

        match_pairs = np.nonzero(iou_mat)
        match_pairs_dict = {}
        if iou_mat.max() > 0:
            for i, pred_id in enumerate(match_pairs[1]):
                if pred_id not in match_pairs_dict.keys():
                    match_pairs_dict[pred_id] = []
                match_pairs_dict[pred_id].append(match_pairs[0][i])
        return match_pairs_dict

    def compute_IOU(self, bbox1, bbox2):
        if isinstance(bbox1['category_id'], str):
            bbox1['category_id'] = int(bbox1['category_id'].replace('\n', ''))
        if isinstance(bbox2['category_id'], str):
            bbox2['category_id'] = int(bbox2['category_id'].replace('\n', ''))
        if bbox1['category_id'] == bbox2['category_id']:
            rec1 = bbox1['bbox']
            rec2 = bbox2['bbox']
            # computing area of each rectangles
            S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
            S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

            # computing the sum_area
            sum_area = S_rec1 + S_rec2

            # find the each edge of intersect rectangle
            left_line = max(rec1[1], rec2[1])
            right_line = min(rec1[3], rec2[3])
            top_line = max(rec1[0], rec2[0])
            bottom_line = min(rec1[2], rec2[2])
            # judge if there is an intersect
            if left_line >= right_line or top_line >= bottom_line:
                return 0
            else:
                intersect = (right_line - left_line) * (bottom_line - top_line)
                return intersect / (sum_area - intersect)
        else:
            return 0

            