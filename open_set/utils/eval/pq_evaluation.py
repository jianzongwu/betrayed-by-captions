#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import numpy as np
from collections import defaultdict
import multiprocessing

import PIL.Image as Image
from panopticapi.utils import get_traceback, rgb2id

OFFSET = 256 * 256 * 256
VOID = 0

class PQStatCat():
        def __init__(self):
            self.iou = 0.0
            self.tp = 0
            self.fp = 0
            self.fn = 0

        def __iadd__(self, pq_stat_cat):
            self.iou += pq_stat_cat.iou
            self.tp += pq_stat_cat.tp
            self.fp += pq_stat_cat.fp
            self.fn += pq_stat_cat.fn
            return self

        def __str__(self):
            return f'iou: {self.iou}, tp: {self.tp}, fp: {self.fp}, fn: {self.fn}'


class PQStat():
    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def pq_average(self, categories, isthing, isunknown, unknown_cat_ids):
        precision, recall, pq, sq, rq, n = 0.0, 0.0, 0, 0, 0, 0
        per_class_results = {}
        for label, label_info in categories.items():
            cat_isthing = label_info['isthing'] == 1
            cat_isunknown = label_info['id'] in unknown_cat_ids
            if isthing is not None:
                if isthing != cat_isthing:
                    continue
            if isunknown is not None:
                if isunknown != cat_isunknown:
                    continue

            iou = self.pq_per_cat[label].iou
            tp_class = self.pq_per_cat[label].tp
            fp_class = self.pq_per_cat[label].fp
            fn_class = self.pq_per_cat[label].fn
            if tp_class + fp_class + fn_class == 0:
                per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0, 'precision': 0.0, 'recall': 0.0}
                continue
            precision_class = tp_class / (tp_class + fp_class) if (tp_class + fp_class) > 0 else 0.0
            recall_class = tp_class / (tp_class + fn_class) if (tp_class + fn_class) > 0 else 0.0
            pq_class = iou / (tp_class + 0.5 * fp_class + 0.5 * fn_class)
            sq_class = iou / tp_class if tp_class != 0 else 0
            rq_class = tp_class / (tp_class + 0.5 * fp_class + 0.5 * fn_class)
            per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class, 'precision': precision_class, 'recall': recall_class}
            n += 1
            precision += precision_class
            recall += recall_class
            pq += pq_class
            sq += sq_class
            rq += rq_class
        
        if n == 0:
            print("There is no class confirmed to the claimed classes!")
            return {'pq': 0, 'sq': 0, 'rq': 0, 'n': 0, 'precision': 0, 'recall': 0}, per_class_results
        
        return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n, 'precision': precision / n, 'recall': recall / n}, per_class_results


@get_traceback
def pq_compute_single_core(proc_id, annotation_set, gt_folder, pred_folder):
    pq_stat = PQStat()

    idx = 0
    for gt_ann, pred_ann in annotation_set:
        # if idx % 1000 == 0:
            # print('Core: {}, {} from {} images processed'.format(proc_id, idx, len(annotation_set)))
        idx += 1

        pan_gt = np.array(Image.open(os.path.join(gt_folder, gt_ann['file_name'])), dtype=np.uint32)
        pan_gt = rgb2id(pan_gt)
        pan_pred = np.array(Image.open(os.path.join(pred_folder, pred_ann['file_name'])), dtype=np.uint32)
        pan_pred = rgb2id(pan_pred)

        gt_segms = {el['id']: el for el in gt_ann['segments_info']}
        pred_segms = {el['id']: el for el in pred_ann['segments_info']}

        # predicted segments area calculation + prediction sanity checks
        pred_labels_set = set(el['id'] for el in pred_ann['segments_info'])
        labels, labels_cnt = np.unique(pan_pred, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label not in pred_segms:
                if label == VOID:
                    continue
                raise KeyError('In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(gt_ann['image_id'], label))
            pred_segms[label]['area'] = label_cnt
            pred_labels_set.remove(label)
            # if pred_segms[label]['category_id'] not in categories:
            #     raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'], label, pred_segms[label]['category_id']))
        if len(pred_labels_set) != 0:
            raise KeyError('In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(gt_ann['image_id'], list(pred_labels_set)))

        # confusion matrix calculation
        pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection

        # count all matched pairs
        gt_matched = set()
        pred_matched = set()
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label not in gt_segms:
                continue
            if pred_label not in pred_segms:
                continue
            if gt_segms[gt_label]['iscrowd'] == 1:
                continue
            elif gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
                continue

            union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)
            iou = intersection / union
            if iou > 0.5:
                pq_stat[gt_segms[gt_label]['category_id']].tp += 1
                pq_stat[gt_segms[gt_label]['category_id']].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)

        # count false negatives
        crowd_labels_dict = {}
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            # crowd segments are ignored
            if gt_info['iscrowd'] == 1:
                crowd_labels_dict[gt_info['category_id']] = gt_label
                continue
            pq_stat[gt_info['category_id']].fn += 1

        # count false positives
        for pred_label, pred_info in pred_segms.items():
            if pred_label in pred_matched:
                continue
            # intersection of the segment with VOID
            intersection = gt_pred_map.get((VOID, pred_label), 0)
            # plus intersection with corresponding CROWD region if it exists
            if pred_info['category_id'] in crowd_labels_dict:
                intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
            # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
            if intersection / pred_info['area'] > 0.5:
                continue
            pq_stat[pred_info['category_id']].fp += 1
    print('Core: {}, all {} images processed'.format(proc_id, len(annotation_set)))
    return pq_stat


def pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder):
    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(pq_compute_single_core,
                                (proc_id, annotation_set, gt_folder, pred_folder))
        processes.append(p)

    pq_stat = PQStat()
    for p in processes:
        pq_stat += p.get()
    return pq_stat
