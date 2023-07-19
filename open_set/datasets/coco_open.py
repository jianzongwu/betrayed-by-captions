# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
import itertools
import logging
import os.path as osp
import tempfile
from unicodedata import category
import warnings
from collections import OrderedDict
import transformers
import clip
import torch
import time

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from mmdet.datasets.api_wrappers import COCO
from ..utils.eval.cocoeval import COCOeval
from ..utils.eval.caption.bleu.bleu import Bleu
from ..utils.eval.caption.cider.cider import Cider
from ..utils.eval.caption.rouge.rouge import Rouge
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

from .utils.parser import LVISParser, NLTKParser, ImageNet21KParser


@DATASETS.register_module()
class CocoDatasetOpen(CustomDataset):

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
               (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
               (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
               (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
               (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
               (134, 134, 103), (145, 148, 174), (255, 208, 186),
               (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
               (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
               (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
               (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
               (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
               (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
               (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
               (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
               (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
               (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
               (191, 162, 208)]

    def __init__(self,
                ann_file,
                pipeline,
                classes=None,
                data_root=None,
                img_prefix='',
                seg_prefix=None,
                proposal_file=None,
                test_mode=False,
                filter_empty_gt=True,
                file_client_args=dict(backend='disk'),
                known_file=None,
                unknown_file=None,
                class_agnostic=False,
                emb_type='bert',
                caption_ann_file=None,
                eval_types=[],
                ann_sample_rate=1.0,
                max_ann_per_image=100,
                nouns_parser='lvis'):
        self.test_mode = test_mode
        self.known_file = known_file
        self.unknown_file = unknown_file
        self.class_agnostic = class_agnostic
        self.file_client_args = file_client_args
        self.caption_ann_file = caption_ann_file
        self.emb_type = emb_type
        self.eval_types = eval_types
        self.ann_sample_rate = ann_sample_rate
        self.max_ann_per_image = max_ann_per_image
        super().__init__(
            ann_file,
            pipeline,
            classes=classes,
            data_root=data_root,
            img_prefix=img_prefix,
            seg_prefix=seg_prefix,
            proposal_file=proposal_file,
            test_mode=test_mode,
            filter_empty_gt=filter_empty_gt,
            file_client_args=file_client_args)

        if self.caption_ann_file is not None:
            self.coco_caption = COCO(self.caption_ann_file)
            self.max_tokens = 35
            self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        if nouns_parser == 'lvis':
            self.parser = LVISParser()
        elif nouns_parser == 'nltk':
            self.parser = NLTKParser()
        elif nouns_parser == 'nouns_adj':
            self.parser = NLTKParser(allowed_tags=['NN', 'NNS', 'JJ', 'JJR', 'JJS'])
        elif nouns_parser == 'obj_nouns_adj':
            self.parser = LVISParser(add_adj=True)
        elif nouns_parser == 'imagenet':
            self.parser = ImageNet21KParser()

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        self.coco_caption = None
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        # known and unknown classes
        self.file_client = mmcv.FileClient(**self.file_client_args)
        self.all_cat_ids = self.cat_ids
        if self.known_file is not None:
            all_cat_names = self.file_client.get_text(self.known_file).split('\n')
            all_cat_ids = self.coco.get_cat_ids(cat_names=all_cat_names)
            self.all_cat_ids = [id for id in self.cat_ids if id in all_cat_ids]
                
        self.unknown_cat_ids = []
        if self.unknown_file is not None:
            unknown_cat_names = self.file_client.get_text(self.unknown_file).split('\n')
            unknown_cat_ids = self.coco.get_cat_ids(cat_names=unknown_cat_names)
            self.unknown_cat_ids = [id for id in self.cat_ids if id in unknown_cat_ids]

        self.known_cat_ids = [id for id in self.cat_ids if id in self.all_cat_ids and id not in self.unknown_cat_ids]

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.known_cat_ids)}

        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            if self.coco_caption is not None:
                caption_ann_ids = self.coco_caption.get_ann_ids(img_ids=[i])
                assert len(caption_ann_ids) > 0, f"All anns should have a caption ann."
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
    
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        data_info = self.data_infos[idx].copy()
        img_id = data_info['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        if self.coco_caption is not None:
            caption_ann_ids = self.coco_caption.get_ann_ids(img_ids=[img_id])
            caption_ann_info = self.coco_caption.load_anns(caption_ann_ids)
            # During training, randomly choose a caption as gt.
            random_idx = np.random.randint(0, len(caption_ann_info))
            caption = caption_ann_info[random_idx]["caption"]
            unique_object_nouns = self.extract_obj(caption)
            data_info["caption"] = caption
            data_info["caption_nouns"] = " ".join(unique_object_nouns)
        return self._parse_ann_info(data_info, ann_info)

    def extract_obj(self, sentence):
        unique_nns = []
        nns, category_ids = self.parser.parse(sentence)
        unique_nns.extend(nns)
        unique_nns = list(set(unique_nns))
        return unique_nns

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _get_valid_imgs(self, valid_cats):
        """Filter images without ground truths corresponding to certian cats."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(valid_cats):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if img_id not in ids_in_cat:
                continue
            valid_inds.append(i)
            valid_img_ids.append(img_id)
        return valid_img_ids

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox, mask, category, caption annotation.

        Args:
            img_info (dict): Information of the image and caption annotaion.
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, caption. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            # unknown has no annotations
            cat_id = ann['category_id']
            if cat_id not in self.all_cat_ids or cat_id in self.unknown_cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                if self.class_agnostic:
                    # 0 for things, 1 for void
                    gt_labels.append(0)
                else:
                    gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        if self.coco_caption is not None:
            padded_ids, attention_mask, padded_nouns_ids, attention_nouns_mask = self.parse_caption(img_info)
        else:
            padded_ids, attention_mask, padded_nouns_ids, attention_nouns_mask = None, None, None, None

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
            caption_ids=padded_ids,
            caption_mask=attention_mask,
            caption_nouns_ids=padded_nouns_ids,
            caption_nouns_mask=attention_nouns_mask)

        return ann

    def parse_caption(self, img_info):
        caption_str = img_info['caption']
        caption_nouns = img_info['caption_nouns']
        padded_ids = [0] * self.max_tokens
        attention_mask = [0] * self.max_tokens
        padded_nouns_ids = [0] * self.max_tokens
        attention_nouns_mask = [0] * self.max_tokens
        if self.emb_type == 'bert':
            caption_ids = self.tokenizer.encode(caption_str, add_special_tokens=True)
            caption_ids = caption_ids[:self.max_tokens]
            padded_ids[:len(caption_ids)] = caption_ids
            attention_mask[:len(caption_ids)] = [1] * len(caption_ids)
            caption_nouns_ids = self.tokenizer.encode(caption_nouns, add_special_tokens=False)
            caption_nouns_ids = caption_nouns_ids[:self.max_tokens]
            padded_nouns_ids[:len(caption_nouns_ids)] = caption_nouns_ids
            attention_nouns_mask[:len(caption_nouns_ids)] = [1] * len(caption_nouns_ids)
        elif self.emb_type == 'clip':
            padded_ids = clip.tokenize(caption_str).numpy()
            attention_mask = padded_ids > 0
            padded_nouns_ids = torch.cat([clip.tokenize(f'A photo of a {noun}') for noun in caption_nouns.split(' ')], dim=0).numpy()
            attention_nouns_mask = [0] * padded_nouns_ids.shape[1]
            attention_nouns_mask[:padded_nouns_ids.shape[0]] = [1] * padded_nouns_ids.shape[0]
        elif self.emb_type == 'bert-clip':
            caption_ids = self.tokenizer.encode(caption_str, add_special_tokens=True)
            caption_ids = caption_ids[:self.max_tokens]
            padded_ids[:len(caption_ids)] = caption_ids
            attention_mask[:len(caption_ids)] = [1] * len(caption_ids)
            padded_nouns_ids = torch.cat([clip.tokenize(f'A photo of a {noun}') for noun in caption_nouns.split(' ')], dim=0).numpy()
            attention_nouns_mask = [0] * padded_nouns_ids.shape[1]
            attention_nouns_mask[:padded_nouns_ids.shape[0]] = [1] * padded_nouns_ids.shape[0]

        return padded_ids, attention_mask, padded_nouns_ids, attention_nouns_mask

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _segm2json(self, results, pred_cats):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = pred_cats[label]
                    if self.class_agnostic:
                        data['isthing'] = label == 0
                    bbox_json_results.append(data)

                # segment results
                segms = seg[label]
                mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = pred_cats[label]
                    if self.class_agnostic:
                        data['isthing'] = label == 0
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix, pred_cats):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".
            pred_cats (list[int]): Prediction label -> cat ids

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        json_results = self._segm2json(results, pred_cats)
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        result_files['segm'] = f'{outfile_prefix}.segm.json'
        mmcv.dump(json_results[0], result_files['bbox'])
        mmcv.dump(json_results[1], result_files['segm'])
        return result_files

    def format_results(self, results, pred_cats, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        result_files = self.results2json(results, jsonfile_prefix, pred_cats)
        return result_files, tmp_dir

    def evaluate_det_segm(self,
                          results,
                          result_files,
                          coco_gt,
                          metrics,
                          logger=None,
                          classwise=False,
                          proposal_nums=(100, 300, 1000),
                          iou_thrs=None,
                          metric_items=None,
                          pred_cats=None):
        """Instance segmentation and object detection evaluation in COCO
        protocol.

        Args:
            results (list[list | tuple | dict]): Testing results of the
                dataset.
            result_files (dict[str, str]): a dict contains json file path.
            coco_gt (COCO): COCO API object with ground truth annotation.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.
            pred_cats (list[int]): Prediction label -> cat ids

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        eval_results = OrderedDict()
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                coco_det = coco_gt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            cocoEval = COCOeval(coco_gt, coco_det, iou_type)
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            if self.class_agnostic:
                cocoEval.params.class_agnostic = True
                print_log('\n' + 'Evaluating in class agnostic mode', logger=logger)
            else:
                cocoEval.params.catIds = pred_cats
                cocoEval.params.imgIds = self._get_valid_imgs(pred_cats)
                print_log('\n' + f'Evaluating {len(pred_cats)} classes with {len(cocoEval.params.imgIds)} images', logger=logger)
                print_log('\n' + f'Categories names: {[self.coco.loadCats(catId)[0]["name"] for catId in pred_cats]}')

            cocoEval.evaluate()
            cocoEval.accumulate()

            if self.known_file:
                print_log('\n' + 'use 48/17 split, do not perform summarize on coco api.')
            else:
                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

            if classwise:  # Compute per-category AR
                for criterion in ['precision']:
                    results = cocoEval.eval[criterion]
                    # recall: (iou, cls, area range, max dets)
                    # precision: (iou, rec_thr, cls, area range, max dets)
                    results_per_category = []
                    results_base = []
                    results_novel = []
                    for idx, catId in enumerate(pred_cats):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]["name"]
                        is_novel = catId in self.unknown_cat_ids
                        is_absent = catId not in self.all_cat_ids
                        if is_absent:
                            nm = '(' + nm + ')'
                        elif is_novel:
                            nm = '*' + nm
                        if criterion == 'recall':
                            result = results[0, idx, 0, -1]
                        elif criterion == 'precision':
                            result = results[0, :, idx, 0, -1]
                        result = result[result > -1]
                        if result.size:
                            avg_result = np.mean(result)
                            if not is_absent:
                                if is_novel:
                                    results_novel.append(avg_result)
                                else:
                                    results_base.append(avg_result)
                        else:
                            avg_result = float('nan')
                        results_per_category.append(
                            (f'{nm}', f'{float(avg_result):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    if criterion == 'recall':
                        headers = ['category', 'AR'] * (num_columns // 2)
                    elif criterion == 'precision':
                        headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                    base_result = np.array(results_base).mean() * 100
                    novel_result = np.array(results_novel).mean() * 100
                    all_result = np.array(results_base + results_novel).mean() * 100
                    print_log('\n' + f'average {criterion}: base {base_result:0.1f}, novel {novel_result:0.1f}, all {all_result:0.1f}', logger=logger)

        return eval_results

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[dict[tuple]]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        coco_gt = self.coco
        self.cat_ids = coco_gt.get_cat_ids(cat_names=self.CLASSES)

        for eval_type in self.eval_types:
            cur_results = [result[eval_type] for result in results]
            pred_cats = self.cat_ids
            if eval_type == 'all_results':
                pred_cats = self.all_cat_ids
            elif eval_type == 'novel_results':
                pred_cats = self.unknown_cat_ids
            elif eval_type == 'base_results':
                pred_cats = self.known_cat_ids
            elif eval_type == 'ins_results':
                pred_cats = self.cat_ids
            elif eval_type == 'visual':
                self.save_results(cur_results)
                continue
            elif eval_type == 'cap_results':
                self.eval_cap_results(cur_results)
                continue

            result_files, tmp_dir = self.format_results(cur_results, pred_cats, jsonfile_prefix)
            eval_results = self.evaluate_det_segm(cur_results, result_files, coco_gt,
                                                metrics, logger, classwise,
                                                proposal_nums, iou_thrs,
                                                metric_items, pred_cats)

            if tmp_dir is not None:
                tmp_dir.cleanup()
                
        return None

    def save_results(self, results):
        embedding_results = []
        cat_results = []
        for idx in range(len(self)):
            embeddings, assigned_labels = results[idx]
            for idx in range(len(embeddings)):
                # bbox results
                embedding = embeddings[idx]
                assigned_label = assigned_labels[idx]
                if assigned_label == len(self.all_cat_ids):
                    continue
                category = self.all_cat_ids[assigned_label]
                embedding_results.append(embedding)
                cat_results.append(category)
        
        embeddings = np.stack(embedding_results, axis=0)
        print(f'embeddings shape: {embeddings.shape}')
        categories = np.array(cat_results)
        print(f'categories.shape: {categories.shape}')

        embedding_path = './results/coco_embedding.npy'
        cat_path = './results/coco_gt_category.npy'

        np.save(embedding_path, embeddings)
        np.save(cat_path, categories)
        print('Visualization results successfully saved!')

    def eval_cap_results(self, results):
        cap_results = {}
        cap_gts = {}
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            # result
            res_sentence = results[idx]
            cap_results[img_id] = [res_sentence]
            # gt
            caption_ann_ids = self.coco_caption.get_ann_ids(img_ids=[img_id])
            caption_ann_info = self.coco_caption.load_anns(caption_ann_ids)
            for caption_ann in caption_ann_info:
                cap_gts[img_id] = cap_gts.get(img_id, []) + [caption_ann['caption']]

        # BLUE
        tic = time.time()
        print(f'start calculating BLUE')
        scorer = Bleu(n=4)
        score, scores = scorer.compute_score(cap_gts, cap_results)
        toc = time.time()
        print(f'time: {(toc - tic):0.2f}')
        for i, s in enumerate(score):
            print(f'BLUE-{i+1} = {s:0.3f}')
        # CIDEr
        tic = time.time()
        print(f'start calculating CIDEr')
        scorer = Cider()
        (score, scores) = scorer.compute_score(cap_gts, cap_results)
        toc = time.time()
        print(f'CIDEr = {score:0.3f} time: {(toc - tic):0.2f}')
        # rouge
        tic = time.time()
        print(f'start calculating Rouge')
        scorer = Rouge()
        score, scores = scorer.compute_score(cap_gts, cap_results)
        toc = time.time()
        print(f'Rouge = {score:0.3f} time: {(toc - tic):0.2f}')