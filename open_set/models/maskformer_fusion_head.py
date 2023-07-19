# Copyright (c) OpenMMLab. All rights reserved.
from typing import OrderedDict

import torch
import torch.nn.functional as F

import mmcv

from mmdet.core.evaluation.panoptic_utils import INSTANCE_OFFSET
from mmdet.core.mask import mask2bbox
from mmdet.models.builder import HEADS
from mmdet.models.seg_heads.panoptic_fusion_heads.base_panoptic_fusion_head import BasePanopticFusionHead


@HEADS.register_module()
class MaskFormerFusionHeadOpen(BasePanopticFusionHead):

    def __init__(self,
                 num_things_classes=65,
                 num_stuff_classes=0,
                 panoptic_mode=False,
                 test_cfg=None,
                 loss_panoptic=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__(num_things_classes, num_stuff_classes, test_cfg,
                         loss_panoptic, init_cfg, **kwargs)
        self.kwargs = kwargs
        self.panoptic_mode = panoptic_mode
        self.use_class_emb = kwargs.get('use_class_emb', False)
        self.known_file = kwargs.get('known_file', None)
        self.unknown_file = kwargs.get('unknown_file', None)

        if self.known_file is not None:
            file_client = mmcv.FileClient()
            self.known_cat_names = file_client.get_text(self.known_file).split('\n')
        if self.unknown_file is not None:
            file_client = mmcv.FileClient()
            self.unknown_cat_names = file_client.get_text(self.unknown_file).split('\n')
        else:
            self.unknown_cat_names = []
        if self.use_class_emb:
            class_to_emb_file = kwargs['class_to_emb_file']
            class_to_emb = mmcv.load(class_to_emb_file)
            ordered_class_names = []
            all_class_embs = torch.zeros((self.num_classes + 1, len(class_to_emb[0]['emb'])), dtype=torch.float)
            novel_class_embs = torch.zeros((len(self.unknown_cat_names) + 1, len(class_to_emb[0]['emb'])), dtype=torch.float)
            base_class_embs = torch.zeros((len(all_class_embs) - len(novel_class_embs) + 1, len(class_to_emb[0]['emb'])), dtype=torch.float)
            i = j = k = 0
            for idx, class_dict in enumerate(class_to_emb):
                if self.known_file:
                    if class_dict['name'] not in self.known_cat_names:
                        continue
                if self.unknown_file:
                    if class_dict['name'] in self.unknown_cat_names:
                        novel_class_embs[j, :] = torch.FloatTensor(class_dict['emb'])
                        j += 1
                    else:
                        base_class_embs[k, :] = torch.FloatTensor(class_dict['emb'])
                        k += 1
                all_class_embs[i, :] = torch.FloatTensor(class_dict['emb'])
                ordered_class_names.append(class_dict['name'])
                i += 1
            # automatically to cuda
            self.register_buffer('all_class_embs', all_class_embs)
            self.register_buffer('novel_class_embs', novel_class_embs)
            self.register_buffer('base_class_embs', base_class_embs)
            self.all_classes = len(all_class_embs) - 1
            self.novel_classes = len(novel_class_embs) - 1
            self.base_classes = len(base_class_embs) - 1
            self.ordered_class_names = ordered_class_names

    def forward_train(self, **kwargs):
        """MaskFormerFusionHead has no training loss."""
        return dict()

    def panoptic_postprocess_emb(self, mask_cls_emb, mask_pred, gt_cls_embs):
        """Panoptic segmengation inference.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_queries, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, h, w) for a image.

        Returns:
            Tensor: Panoptic segment result of shape \
                (h, w), each element in Tensor means: \
                ``segment_id = _cls + instance_id * INSTANCE_OFFSET``.
        """
        object_mask_thr = self.test_cfg.get('object_mask_thr', 0.8)
        iou_thr = self.test_cfg.get('iou_thr', 0.8)
        filter_low_score = self.test_cfg.get('filter_low_score', False)
        stuff_area_limit = self.test_cfg.get('stuff_area_limit', 4096)
        mask_cls_emb_score = self.get_cls_emb_scores(mask_cls_emb, gt_cls_embs)

        scores, labels = mask_cls_emb_score.max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.num_classes) & (scores > object_mask_thr)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        # num_classes as background
        panoptic_seg = torch.full((h, w),
                                  self.num_classes,
                                  dtype=torch.int32,
                                  device=cur_masks.device)
        stuff_query_list = []
        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            pass
        else:
            cur_mask_ids = cur_prob_masks.argmax(0)
            instance_id = 1
            for k in range(cur_classes.shape[0]):
                pred_class = int(cur_classes[k].item())
                isthing = pred_class < self.num_things_classes
                mask = cur_mask_ids == k
                mask_area = mask.sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()

                if filter_low_score:
                    mask = mask & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0:
                    if mask_area / original_area < iou_thr:
                        continue

                    if not isthing:
                        # different stuff regions of same class will be
                        # merged here, and stuff share the instance_id 0.
                        # panoptic_seg[mask] = pred_class
                        # ignore the stuff and index stuff query
                        stuff_query_list.append(k)
                        continue
                    else:
                        panoptic_seg[mask] = (
                            pred_class + instance_id * INSTANCE_OFFSET)
                        instance_id += 1

            # to paste the stuff
            for k in stuff_query_list:
                pred_class = int(cur_classes[k].item())
                mask = cur_mask_ids == k
                # fill the remaining background classes
                mask = mask & (panoptic_seg == self.num_classes)
                mask_area = mask.sum().item()
                if mask_area < stuff_area_limit:
                    continue
                panoptic_seg[mask] = pred_class

        return panoptic_seg

    def panoptic_postprocess(self, mask_cls, mask_pred):
        """Panoptic segmengation inference.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_queries, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, h, w) for a image.

        Returns:
            Tensor: Panoptic segment result of shape \
                (h, w), each element in Tensor means: \
                ``segment_id = _cls + instance_id * INSTANCE_OFFSET``.
        """
        object_mask_thr = self.test_cfg.get('object_mask_thr', 0.8)
        iou_thr = self.test_cfg.get('iou_thr', 0.8)
        filter_low_score = self.test_cfg.get('filter_low_score', False)

        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.num_classes) & (scores > object_mask_thr)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.full((h, w),
                                  self.num_classes,
                                  dtype=torch.int32,
                                  device=cur_masks.device)
        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            pass
        else:
            cur_mask_ids = cur_prob_masks.argmax(0)
            instance_id = 1
            for k in range(cur_classes.shape[0]):
                pred_class = int(cur_classes[k].item())
                isthing = pred_class < self.num_things_classes
                mask = cur_mask_ids == k
                mask_area = mask.sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()

                if filter_low_score:
                    mask = mask & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0:
                    if mask_area / original_area < iou_thr:
                        continue

                    if not isthing:
                        # different stuff regions of same class will be
                        # merged here, and stuff share the instance_id 0.
                        panoptic_seg[mask] = pred_class
                    else:
                        panoptic_seg[mask] = (
                            pred_class + instance_id * INSTANCE_OFFSET)
                        instance_id += 1

        return panoptic_seg

    def semantic_postprocess(self, mask_cls, mask_pred):
        """Semantic segmengation postprocess.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_queries, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, h, w) for a image.

        Returns:
            Tensor: Semantic segment result of shape \
                (cls_out_channels, h, w).
        """
        # TODO add semantic segmentation result
        raise NotImplementedError

    def instance_postprocess(self, mask_cls, mask_pred):
        """Instance segmengation postprocess.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_queries, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, h, w) for a image.

        Returns:
            tuple[Tensor]: Instance segmentation results.

            - labels_per_image (Tensor): Predicted labels,\
                shape (n, ).
            - bboxes (Tensor): Bboxes and scores with shape (n, 5) of \
                positive region in binary mask, the last column is scores.
            - mask_pred_binary (Tensor): Instance masks of \
                shape (n, h, w).
        """
        max_per_image = self.test_cfg.get('max_per_image', 100)
        num_queries = mask_cls.shape[0]
        # shape (num_queries, num_class)
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        # shape (num_queries * num_class, )
        labels = torch.arange(self.num_classes, device=mask_cls.device).\
            unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
        scores_per_image, top_indices = scores.flatten(0, 1).topk(
            max_per_image, sorted=False)
        labels_per_image = labels[top_indices]

        query_indices = top_indices // self.num_classes
        mask_pred = mask_pred[query_indices]

        # extract things
        is_thing = labels_per_image < self.num_things_classes
        scores_per_image = scores_per_image[is_thing]
        labels_per_image = labels_per_image[is_thing]
        mask_pred = mask_pred[is_thing]

        mask_pred_binary = (mask_pred > 0).float()
        mask_scores_per_image = (mask_pred.sigmoid() *
                                 mask_pred_binary).flatten(1).sum(1) / (
                                     mask_pred_binary.flatten(1).sum(1) + 1e-6)
        det_scores = scores_per_image * mask_scores_per_image
        mask_pred_binary = mask_pred_binary.bool()
        bboxes = mask2bbox(mask_pred_binary)
        bboxes = torch.cat([bboxes, det_scores[:, None]], dim=-1)

        return labels_per_image, bboxes, mask_pred_binary

    def get_cls_emb_scores(self, cls_emb_preds, gt_cls_embs):
        """Compute scores for embedding prediction head.

        Args:
            cls_emb_preds (Tensor): Embedding prediction for a single decoder
                layer for all images. Shape
                (num_queries, d_l).
            gt_cls_embs (Tensor): (n, d_l)
            
        Returns:
            cls_emb_scores (Tensor): Embedding predicion scores.
                (num_queries, self.num_classes + 1).
        """
        # (num_queries, d_l) * (d_l, self.num_classes) ->
        # (num_queries, self.num_classes)
        dot_products = torch.matmul(cls_emb_preds, gt_cls_embs.t())
        cls_emb_scores = F.softmax(dot_products, -1)

        return cls_emb_scores

    def instance_postprocess_emb(self, mask_cls_emb, mask_pred, gt_cls_embs):
        """Instance segmengation postprocess.

        Args:
            mask_cls_emb (Tensor): Embedding prediction of shape
                (num_queries, d_l) for a image.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, h, w) for a image.

        Returns:
            tuple[Tensor]: Instance segmentation results.

            - names_per_image (list[str]): Predicted class names of length n.
            - bboxes (Tensor): Bboxes and scores with shape (n, 5) of \
                positive region in binary mask, the last column is scores.
            - mask_pred_binary (Tensor): Instance masks of \
                shape (n, h, w).
        """
        max_per_image = self.test_cfg.get('max_per_image', 100)
        num_queries = mask_cls_emb.shape[0]
        # (num_queries, self.num_classes + 1)
        mask_cls_emb_score = self.get_cls_emb_scores(mask_cls_emb, gt_cls_embs)
        # shape (num_queries, num_class)
        scores = mask_cls_emb_score[:, :-1]
        # shape (num_queries * num_class, )
        labels = torch.arange(scores.shape[-1], device=mask_cls_emb_score.device).\
            unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
        scores_per_image, top_indices = scores.flatten(0, 1).topk(
            max_per_image, sorted=False)
        labels_per_image = labels[top_indices]

        query_indices = top_indices // scores.shape[-1]
        mask_pred = mask_pred[query_indices]

        # extract non-void
        is_valid = labels_per_image < scores.shape[-1]
        scores_per_image = scores_per_image[is_valid]
        labels_per_image = labels_per_image[is_valid]
        mask_pred = mask_pred[is_valid]

        mask_pred_binary = (mask_pred > 0).float()
        mask_scores_per_image = (mask_pred.sigmoid() *
                                 mask_pred_binary).flatten(1).sum(1) / (
                                     mask_pred_binary.flatten(1).sum(1) + 1e-6)
        det_scores = scores_per_image * mask_scores_per_image
        mask_pred_binary = mask_pred_binary.bool()
        bboxes = mask2bbox(mask_pred_binary)
        bboxes = torch.cat([bboxes, det_scores[:, None]], dim=-1)

        return labels_per_image, bboxes, mask_pred_binary


    def simple_test(self,
                    mask_cls_results,
                    mask_cls_emb_results,
                    mask_pred_results,
                    img_metas,
                    **kwargs):
        """Test segment without test-time aumengtation.

        Only the output of last decoder layers was used.

        Args:
            mask_cls_results (Tensor): Mask classification logits,
                shape (batch_size, num_queries, cls_out_channels).
                Note `cls_out_channels` should includes background.
            mask_cls_emb_results (Tensor): Embedding prediction,
                shape (batch_size, num_queries, d_l)
            mask_pred_results (Tensor): Mask logits, shape
                (batch_size, num_queries, h, w).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): If True, return boxes in
                original image space. Default False.

        Returns:
            list[dict[str, Tensor | tuple[Tensor]]]: Semantic segmentation \
                results and panoptic segmentation results for each \
                image.

            .. code-block:: none

                [
                    {
                        'pan_results': Tensor, # shape = [h, w]
                        'ins_results': tuple[Tensor],
                        # semantic segmentation results are not supported yet
                        'sem_results': Tensor
                    },
                    ...
                ]
        """
        eval_types = self.test_cfg.get('eval_types', [])

        rescale = kwargs.get('rescale', False)
        results = []
        for mask_cls_result, mask_cls_emb_result, mask_pred_result, meta in zip(
                mask_cls_results, mask_cls_emb_results, mask_pred_results, img_metas):
            # remove padding
            img_height, img_width = meta['img_shape'][:2]
            mask_pred_result = mask_pred_result[:, :img_height, :img_width]

            if rescale:
                # return result in original resolution
                ori_height, ori_width = meta['ori_shape'][:2]
                mask_pred_result = F.interpolate(
                    mask_pred_result[:, None],
                    size=(ori_height, ori_width),
                    mode='bilinear',
                    align_corners=False)[:, 0]

            result = dict()

            if 'all_results' in eval_types:
                # panoptic mode
                if self.panoptic_mode:
                    all_results = self.panoptic_postprocess_emb(
                        mask_cls_emb_result, mask_pred_result, self.all_class_embs)
                    result['panoptic_all_results'] = all_results
                else:
                    all_results = self.instance_postprocess_emb(
                    mask_cls_emb_result, mask_pred_result, self.all_class_embs)

                    result['all_results'] = all_results

            if 'novel_results' in eval_types:
                novel_results = self.instance_postprocess_emb(
                    mask_cls_emb_result, mask_pred_result, self.novel_class_embs)

                result['novel_results'] = novel_results

            if 'base_results' in eval_types:
                base_results = self.instance_postprocess_emb(
                    mask_cls_emb_result, mask_pred_result, self.base_class_embs)
                result['base_results'] = base_results

            if 'ins_results' in eval_types:
                ins_results = self.instance_postprocess(
                    mask_cls_result, mask_pred_result)
                result['ins_results'] = ins_results\
            
            if 'pan_results' in eval_types:
                pan_results = self.panoptic_postprocess(
                    mask_cls_result, mask_pred_result)
                result['pan_results'] = pan_results

            results.append(result)

        return results
