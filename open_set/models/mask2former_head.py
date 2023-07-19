# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
import clip

import mmcv
from mmcv.cnn import Conv2d, build_plugin_layer, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.ops import point_sample
from mmcv.runner import ModuleList, force_fp32, get_dist_info

from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.models.utils import preprocess_panoptic_gt
from mmdet.models.utils import get_uncertain_point_coords_with_randomness
from mmdet.models.builder import HEADS, build_loss, build_head
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.dense_heads.maskformer_head import MaskFormerHead

from ..utils.eval.inference import beam_search, get_ids_embedding
from .utils.bert_embeddings import BertEmbeddings

BOS_TOKEN = 101
EOS_TOKEN = 102

@HEADS.register_module()
class Mask2FormerHeadOpen(MaskFormerHead):
    """Implements the Mask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`mmcv.ConfigDict` | dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder position encoding. Defaults to None.
        loss_cls (:obj:`mmcv.ConfigDict` | dict): Config of the classification
            loss. Defaults to None.
        loss_mask (:obj:`mmcv.ConfigDict` | dict): Config of the mask loss.
            Defaults to None.
        loss_dice (:obj:`mmcv.ConfigDict` | dict): Config of the dice loss.
            Defaults to None.
        train_cfg (:obj:`mmcv.ConfigDict` | dict): Training config of
            Mask2Former head.
        test_cfg (:obj:`mmcv.ConfigDict` | dict): Testing config of
            Mask2Former head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
        void_loss (str): Masks not matched to gts are seen as void masks.
            Defaults to 'void-background'.
            'void-background': train label of void masks as background.
            'void-thing': void as thing.
            'void-suppression': negative supervision to the void regions predicted as stuff.
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 num_things_classes=80, # num_known_classes
                 num_stuff_classes=53,
                 num_queries=100,
                 num_transformer_feat_level=3,
                 pixel_decoder=None,
                 enforce_decoder_input_project=False,
                 transformer_decoder=None,
                 positional_encoding=None,
                 v2l_head=None,
                 caption_generator=None,
                 loss_cls=None,
                 loss_cls_emb=None,
                 loss_grounding=None,
                 loss_caption_generation=None,
                 loss_caption_align=None,
                 loss_mask=None,
                 loss_dice=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.transformerlayers.\
            attn_cfgs.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        assert pixel_decoder.encoder.transformerlayers.\
            attn_cfgs.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels)
        self.pixel_decoder = build_plugin_layer(pixel_decoder_)[1]
        self.transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv2d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = build_positional_encoding(
            positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level,
                                        feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))
        self.feat_channels = feat_channels

        self.v2l_head_cfg = v2l_head
        self.caption_generator_cfg = caption_generator

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            self.sampler = build_sampler(self.train_cfg.sampler, context=self)
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.class_weight = loss_cls.class_weight
        self.loss_cls = build_loss(loss_cls)
        if loss_cls_emb is not None:
            self.loss_cls_emb = build_loss(loss_cls_emb)
        if loss_grounding is not None:
            self.loss_grounding = build_loss(loss_grounding)
        if loss_caption_generation is not None:
            self.loss_caption_generation = build_loss(loss_caption_generation)
        if loss_caption_align is not None:
            self.loss_caption_align = build_loss(loss_caption_align)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)

        self.init_kwargs(**kwargs)

    def init_kwargs(self, **kwargs):
        self.kwargs = kwargs
        self.class_agnostic = kwargs.get('class_agnostic', False)
        self.use_class_emb = kwargs.get('use_class_emb', False)
        self.use_caption = kwargs.get('use_caption', False)
        self.use_caption_generation = kwargs.get('use_caption_generation', False)
        self.use_caption_align = kwargs.get('use_caption_align', False)
        self.known_file = kwargs.get('known_file', None)
        self.unknown_file = kwargs.get('unknown_file', None)
        self.softmax_temperature = kwargs.get('softmax_temperature', 10.0)
        self.learnable_temperature = kwargs.get('learnable_temperature', False)
        self.pred_emb_norm = kwargs.get('pred_emb_norm', False)
        self.text_emb_norm = kwargs.get('text_emb_norm', True)
        self.freeze_pretrained = kwargs.get('freeze_pretrained', False)
        self.freeze_v2l = kwargs.get('freeze_v2l', False)
        self.loss_only_last = kwargs.get('loss_only_last', False)
        self.loss_aux_weight = kwargs.get('loss_aux_weight', 1.0)
        self.gen_only_obj_nouns = kwargs.get('gen_only_obj_nouns', False)
        self.gen_mask_obj_nouns = kwargs.get('gen_mask_obj_nouns', False)
        self.gen_replace_obj_nouns = kwargs.get('gen_replace_obj_nouns', False)

        if self.known_file is not None:
            file_client = mmcv.FileClient()
            self.known_cat_names = file_client.get_text(self.known_file).split('\n')
        if self.unknown_file is not None:
            file_client = mmcv.FileClient()
            self.unknown_cat_names = file_client.get_text(self.unknown_file).split('\n')
        if self.use_class_emb:
            class_to_emb_file = kwargs['class_to_emb_file']
            class_to_emb = mmcv.load(class_to_emb_file)
            class_embs = torch.zeros((self.num_classes + 1, len(class_to_emb[0]['emb'])), dtype=torch.float)
            i = 0
            for class_dict in class_to_emb:
                if self.known_file:
                    if class_dict['name'] not in self.known_cat_names:
                        continue
                if self.unknown_file:
                    if class_dict['name'] in self.unknown_cat_names:
                        continue
                class_embs[i, :] = torch.FloatTensor(class_dict['emb'])
                i += 1
            # automatically to cuda
            self.register_buffer('class_embs', class_embs)
            # self.v2l_transform = build_head(self.v2l_head_cfg)
            self.v2l_transform = nn.Linear(self.feat_channels, class_embs.shape[1])
        self.bert_embeddings = self.clip = None
        if self.use_caption:
            self.caption_emb_type = kwargs.get('caption_emb_type', 'clip')
            self.build_text_encoders(self.caption_emb_type)
        if self.use_caption_generation:
            self.caption_gen_emb_type = kwargs.get('caption_gen_emb_type', 'bert')
            self.caption_generator = build_head(self.caption_generator_cfg)
            self.build_text_encoders(self.caption_gen_emb_type)
        if self.learnable_temperature:
            self.softmax_temperature = nn.Parameter(torch.tensor([self.softmax_temperature]), requires_grad=True)

    def init_weights(self):
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        if self.freeze_v2l:
            for p in self.v2l_transform.parameters():
                p.requires_grad = False
        
        if self.freeze_pretrained:
            self.freeze_params()

    def build_text_encoders(self, emb_type):
        if emb_type == 'bert' and self.bert_embeddings is None:
            bert_model = transformers.BertModel.from_pretrained('bert-base-uncased').eval()
            self.bert_embeddings = BertEmbeddings(bert_model)
            for param in self.bert_embeddings.parameters():
                param.requires_grad = False
        if emb_type == 'clip' and self.clip is None:
            # clip_model, _ = clip.load('ViT-B/32')
            clip_model, _ = clip.load('RN50')
            self.clip = clip_model.eval()
            for param in self.clip.parameters():
                param.requires_grad = False

    def freeze_params(self):
        self.decoder_input_projs.eval()
        self.pixel_decoder.eval()
        self.transformer_decoder.eval()
        for p in self.decoder_input_projs.parameters():
            p.requires_grad = False
        for p in self.pixel_decoder.parameters():
            p.requires_grad = False
        for p in self.transformer_decoder.parameters():
            p.requires_grad = False

    def get_targets(self, cls_scores_list, cls_emb_logits_list, mask_preds_list,
                    gt_labels_list, gt_masks_list, img_metas):
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape (num_queries,
                cls_out_channels).
            cls_emb_logits_list: (list[Tensor]): Embedding prediction logits for a single decoder
                layer for all images with shape (num_queries, cls_out_channels).
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape (num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for all
                images. Each with shape (n, ), n is the sum of number of stuff
                type and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[list[Tensor]]: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels of all images.\
                    Each with shape (num_queries, ).
                - label_weights_list (list[Tensor]): Label weights\
                    of all images. Each with shape (num_queries, ).
                - mask_targets_list (list[Tensor]): Mask targets of\
                    all images. Each with shape (num_queries, h, w).
                - mask_weights_list (list[Tensor]): Mask weights of\
                    all images. Each with shape (num_queries, ).
                - num_total_pos (int): Number of positive samples in\
                    all images.
                - num_total_neg (int): Number of negative samples in\
                    all images.
        """
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list,
                                      cls_emb_logits_list, mask_preds_list, gt_labels_list,
                                      gt_masks_list, img_metas)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, mask_targets_list,
                mask_weights_list, num_total_pos, num_total_neg)


    def _get_target_single(self, cls_score, cls_emb_logit, mask_pred,
                           gt_labels, gt_masks, img_metas):
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            cls_emb_logit (Tensor): Embedding prediction logit for a single decoder
                layer for one image with shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, h, w).
            gt_labels (Tensor): Ground truth class indices for one image with
                shape (num_gts, ).
            gt_masks (Tensor): Ground truth mask for each image, each with
                shape (num_gts, h, w).  
            img_metas (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
        """
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)
        # shape (num_queries, num_points)
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1,
                                                        1)).squeeze(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1,
                                                               1)).squeeze(1)

        # assign and sample
        assign_result = self.assigner.assign(cls_score, cls_emb_logit, mask_points_pred,
                                             gt_labels, gt_points_masks,
                                             img_metas)
        sampling_result = self.sampler.sample(assign_result, mask_pred,
                                              gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((self.num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries, ))

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds)

    @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds'))
    def loss(self, all_cls_scores, all_cls_emb_preds, all_mask_preds, 
            gt_labels_list, gt_masks_list, gt_caption_ids_list,
            gt_caption_embs_list, gt_caption_mask_list,
            gt_caption_nouns_ids_list, gt_caption_nouns_embs_list, gt_caption_nouns_mask_list, img_metas):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape (num_decoder, batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            all_cls_emb_preds (Tensor): Embedding prediction for all decoder
                layers with shape (batch_size, num_queries, d_l).
                d_l is the dimension of embeddings.
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape (num_decoder, batch_size, num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (n, ). n is the sum of number of stuff type
                and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image with
                shape (n, h, w).
            gt_caption_ids_list (list[Tensor]): (max_token,)
            gt_caption_embs_list (list[Tensor]): (max_token, d_l)
            gt_caption_mask_list (list[Tensor]): (max_token,)
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_caption_ids_list = [gt_caption_ids_list for _ in range(num_dec_layers)]
        all_gt_caption_embs_list = [gt_caption_embs_list for _ in range(num_dec_layers)]
        all_gt_caption_mask_list = [gt_caption_mask_list for _ in range(num_dec_layers)]
        all_gt_caption_nouns_ids_list = [gt_caption_nouns_ids_list for _ in range(num_dec_layers)]
        all_gt_caption_nouns_embs_list = [gt_caption_nouns_embs_list for _ in range(num_dec_layers)]
        all_gt_caption_nouns_mask_list = [gt_caption_nouns_mask_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_cls_emb, losses_grounding, losses_caption_generation, losses_caption_align, losses_mask, losses_dice = multi_apply(
            self.loss_single, all_cls_scores, all_cls_emb_preds, all_mask_preds,
            all_gt_labels_list, all_gt_masks_list, all_gt_caption_ids_list,
            all_gt_caption_embs_list, all_gt_caption_mask_list,
            all_gt_caption_nouns_ids_list, all_gt_caption_nouns_embs_list, all_gt_caption_nouns_mask_list, img_metas_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_cls_emb'] = losses_cls_emb[-1]
        loss_dict['loss_grounding'] = losses_grounding[-1]
        loss_dict['loss_caption_generation'] = losses_caption_generation[-1]
        loss_dict['loss_caption_align'] = losses_caption_align[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        if self.loss_only_last:
            return loss_dict
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_cls_emb_i, loss_grounding_i, losses_caption_generation_i, losses_caption_align_i, loss_mask_i, loss_dice_i in zip(
                losses_cls[:-1], losses_cls_emb[:-1], losses_grounding[:-1], losses_caption_generation[:-1], losses_caption_align[:-1], losses_mask[:-1], losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i * self.loss_aux_weight
            loss_dict[f'd{num_dec_layer}.loss_cls_emb'] = loss_cls_emb_i * self.loss_aux_weight
            loss_dict[f'd{num_dec_layer}.loss_grounding'] = loss_grounding_i * self.loss_aux_weight
            loss_dict[f'd{num_dec_layer}.loss_caption_generation'] = losses_caption_generation_i * self.loss_aux_weight
            loss_dict[f'd{num_dec_layer}.loss_caption_align'] = losses_caption_align_i * self.loss_aux_weight
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i * self.loss_aux_weight
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i * self.loss_aux_weight
            num_dec_layer += 1
        return loss_dict

    def loss_single(self, cls_scores, cls_emb_preds, mask_preds,
                    gt_labels_list, gt_masks_list,
                    gt_caption_ids_list, gt_caption_embs_list, gt_caption_mask_list,
                    gt_caption_nouns_ids_list, gt_caption_nouns_embs_list, gt_caption_nouns_mask_list, img_metas):
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            cls_emb_preds (Tensor): Embedding prediction for a single decoder
                layer for all images with shape (batch_size, num_queries, d_l).
                d_l is the dimension of embeddings.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image, each with shape (num_gts, ).
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (num_gts, h, w).
            gt_caption_ids_list (list[Tensor]): (max_token,)
            gt_caption_embs_list (list[Tensor]): (max_token, d_l)
            gt_caption_mask_list (list[Tensor]): (max_token,)
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        if self.use_class_emb:
            cls_emb_logits = self._get_cls_emb_logits(cls_emb_preds)
            cls_emb_logits_list = [cls_emb_logits[i] for i in range(num_imgs)]
        else:
            cls_emb_logits_list = [None for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list, 
        num_total_pos, num_total_neg) = self.get_targets(cls_scores_list, cls_emb_logits_list, mask_preds_list,
                                           gt_labels_list, gt_masks_list,
                                           img_metas)

        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        # embedding prediction loss
        loss_cls_emb = loss_cls.new_tensor(0.0)
        if self.use_class_emb:
            cls_emb_logits = cls_emb_logits.flatten(0, 1)
            loss_cls_emb = self.loss_cls_emb(
                cls_emb_logits,
                labels,
                label_weights.float(),
                avg_factor=class_weight[labels].sum())

        # caption grounding loss
        loss_grounding = loss_cls.new_tensor(0.0)
        if self.use_caption:
            all_gt_caption_nouns_embs, all_gt_caption_nouns_mask, all_cls_emb_preds = \
                self.gather_captions_and_preds(gt_caption_nouns_embs_list, gt_caption_nouns_mask_list, cls_emb_preds)
            loss_grounding = self.loss_grounding(
                all_cls_emb_preds,
                all_gt_caption_nouns_embs,
                all_gt_caption_nouns_mask,
                self.softmax_temperature)

        # caption generation loss
        loss_caption_generation = loss_cls.new_tensor(0.0)
        if self.use_caption_generation:
            gt_caption_embs = torch.stack(gt_caption_embs_list, dim=0)
            gt_caption_masks = torch.stack(gt_caption_mask_list, dim=0).bool()
            caption_logits = self.caption_generator(
                tgt=gt_caption_embs[:, :-1, :],
                memory=cls_emb_preds,
                tgt_key_padding_mask=torch.logical_not(gt_caption_masks[:, :-1]))[1]
            # (batch_size * (max_tokens - 1), vocab_size)
            caption_logits = caption_logits.flatten(0, 1)
            for i in range(len(gt_caption_ids_list)):
                gt_caption_ids =  gt_caption_ids_list[i]
                gt_caption_nouns_ids = gt_caption_nouns_ids_list[i].cpu().numpy().tolist()
                for j in range(len(gt_caption_ids)):
                    if int(gt_caption_ids[j]) not in gt_caption_nouns_ids:
                        if self.gen_only_obj_nouns:
                            # set gt to 0 except for obj nouns
                            gt_caption_ids[j] = 0
                    elif int(gt_caption_ids[j]) in gt_caption_nouns_ids:
                        if self.gen_mask_obj_nouns:
                            # set 0 to one object noun (the first one seen in the caption)
                            gt_caption_ids[j] = 0
                            break
                        if self.gen_replace_obj_nouns:
                            gt_caption_ids[j] = 4874    # 'object'
            # (batch_size * (max_tokens - 1))
            gt_caption_ids = torch.stack(gt_caption_ids_list, dim=0)[:, 1:].flatten(0, 1)
            loss_caption_generation = self.loss_caption_generation(
                caption_logits,
                gt_caption_ids)
            
        loss_caption_align = loss_cls.new_tensor(0.0)
        if self.use_caption_align:
            gt_caption_nouns_embs = torch.stack(gt_caption_nouns_embs_list, dim=0)
            gt_caption_nouns_masks = torch.stack(gt_caption_nouns_mask_list, dim=0).bool()
            loss_caption_align = self.loss_caption_align(
                cls_emb_preds,
                gt_caption_nouns_embs,
                gt_caption_nouns_masks)

        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_cls_emb, loss_grounding, loss_caption_generation, loss_caption_align, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)

        return loss_cls, loss_cls_emb, loss_grounding, loss_caption_generation, loss_caption_align, loss_mask, loss_dice 

    def _get_cls_emb_logits(self, cls_emb_preds):
        """Compute prediction logits for embedding predicion head. 

        Args:
            cls_emb_preds (Tensor): Embedding prediction for a single decoder
                layer for all images. Shape
                (batch_size, num_queries, d_l).
            
        Returns:
            cls_emb_logits (Tensor): Embedding predicion scores.
                (batch_size, num_queries, self.num_classes + 1).
        """
        # (batch_size, num_queries, d_l) * (d_l, self.num_classes) ->
        # (batch_size, num_queries, self.num_classes)
        logits = torch.matmul(cls_emb_preds, self.class_embs.t()) / self.softmax_temperature
        # print("logits, ", logits.shape)

        return logits

    def gather_captions_and_preds(self, gt_caption_embs_list, gt_caption_mask_list, cls_emb_preds):
        """Gather all caption annotations from the whole batch using dist.all_gather. 

        Args:
            gt_caption_embs_list (list[Tensor]): (max_token, d_l)
            gt_caption_mask_list (list[Tensor]): (max_token)
            cls_emb_preds (Tensor): (batch_size, num_queries, d_l).
        Returns:
            all_gt_caption_embs (Tensor): (batch_size * world_size, max_tokens, d_l)
            all_gt_caption_mask (Tensor): (batch_size * world_size, max_tokens).
            all_cls_emb_preds (Tensor): (batch_size * world_size, num_queries, d_l).     
        """
        batch_size = len(gt_caption_embs_list)
        rank, world_size = get_dist_info()
        if world_size > 1:
            gt_caption_embs = torch.stack(gt_caption_embs_list, dim=0)
            gt_caption_mask = torch.stack(gt_caption_mask_list, dim=0)
            emb_tmp_list = [gt_caption_embs.new_zeros(size=gt_caption_embs.size()) for i in range(world_size)]
            mask_tmp_list = [gt_caption_mask.new_zeros(size=gt_caption_mask.size()) for i in range(world_size)]
            pred_tmp_list = [cls_emb_preds.new_zeros(size=cls_emb_preds.size()) for i in range(world_size)]
        
            dist.all_gather(emb_tmp_list, gt_caption_embs)
            dist.all_gather(mask_tmp_list, gt_caption_mask)
            dist.all_gather(pred_tmp_list, cls_emb_preds)

            all_gt_caption_embs = torch.cat(emb_tmp_list, dim=0)
            all_gt_caption_mask = torch.cat(mask_tmp_list, dim=0)
            all_cls_emb_preds = torch.cat(pred_tmp_list, dim=0)
            all_cls_emb_preds[rank * batch_size : (rank + 1) * batch_size] = cls_emb_preds
            return all_gt_caption_embs, all_gt_caption_mask, all_cls_emb_preds
        else:
            all_gt_caption_embs = torch.stack(gt_caption_embs_list, dim=0)
            all_gt_caption_mask = torch.stack(gt_caption_mask_list, dim=0)
            all_cls_emb_preds = cls_emb_preds
            return all_gt_caption_embs, all_gt_caption_mask, all_cls_emb_preds

    def extract_word_embeddings(self, ids_list, mask_list, emb_type='bert'):
        """extract caption words' embeddings and masks
        """

        embs_list = []
        valid_mask_list = []
        if emb_type == 'bert':
            for i, ids in enumerate(ids_list):
                embs = self.bert_embeddings.word_embeddings(ids)
                if self.text_emb_norm:
                    embs = self.bert_embeddings.LayerNorm(embs)
                embs_list.append(embs)
                valid_mask_list.append(mask_list[i])
        elif emb_type == 'clip':
            for i, ids in enumerate(ids_list):
                nouns_embs = self.clip.encode_text(ids).float()
                if self.text_emb_norm:
                    nouns_embs /= nouns_embs.norm(dim=-1, keepdim=True)
                embs = nouns_embs.new_zeros((ids.shape[1], nouns_embs.shape[-1]))
                embs[:nouns_embs.shape[0], :] = nouns_embs
                embs_list.append(embs)
                valid_mask_list.append(mask_list[i])

        return embs_list, valid_mask_list

    def forward_head(self, decoder_out, mask_feature, attn_mask_target_size):
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (num_queries, batch_size, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

            - cls_pred (Tensor): Classification scores in shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - cls_emb_pred (Tensor): Embedding prediction in shape \
                (batch_size, num_queries, d_l). \
                d_l is the dimension of embeddings.
            - mask_pred (Tensor): Mask scores in shape \
                (batch_size, num_queries,h, w).
            - attn_mask (Tensor): Attention mask in shape \
                (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)
        # shape (num_queries, batch_size, c)
        cls_pred = self.cls_embed(decoder_out)
        # shape (num_queries, batch_size, d_l)
        cls_emb_pred = cls_pred
        if self.use_class_emb:
            cls_emb_pred = self.v2l_transform(decoder_out)
            # normalization the embedding prediction
            if self.pred_emb_norm:
                cls_emb_pred = cls_emb_pred / cls_emb_pred.norm(dim=-1, keepdim=True)
        # shape (num_queries, batch_size, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, cls_emb_pred, mask_pred, attn_mask

    def forward(self, feats, img_metas):
        """Forward function.

        Args:
            feats (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two elements.

            - cls_pred_list (list[Tensor)]: Classification logits \
                for each decoder layer. Each is a 3D-tensor with shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - cls_emb_pred_list (list[Tensor]): Embedding prediction \
                for each decoder layer. Each is a 3D-tensor with shape
                (batch_size, num_queries, d_l). \
                d_l is the dimension of embeddings.
            - mask_pred_list (list[Tensor]): Mask logits for each \
                decoder layer. Each with shape (batch_size, num_queries, \
                 h, w).
        """
        batch_size = len(img_metas)
        mask_features, multi_scale_memorys = self.pixel_decoder(feats)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(2, 0, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (num_queries, batch_size, c)
        query_feat = self.query_feat.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))

        cls_pred_list = []
        cls_emb_pred_list = []
        mask_pred_list = []
        cls_pred, cls_emb_pred, mask_pred, attn_mask = self.forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        cls_emb_pred_list.append(cls_emb_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(
                attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            attn_masks = [attn_mask, None]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, cls_emb_pred, mask_pred, attn_mask = self.forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            cls_emb_pred_list.append(cls_emb_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, cls_emb_pred_list, mask_pred_list

    def forward_train(self,
                      feats,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_semantic_seg,
                      gt_caption_ids,
                      gt_caption_mask,
                      gt_caption_nouns_ids,
                      gt_caption_nouns_mask,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward function for training mode.

        Args:
            feats (list[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            img_metas (list[Dict]): List of image information.
            gt_bboxes (list[Tensor]): Each element is ground truth bboxes of
                the image, shape (num_gts, 4). Not used here.
            gt_labels (list[Tensor]): Each element is ground truth labels of
                each box, shape (num_gts,).
            gt_masks (list[BitmapMasks]): Each element is masks of instances
                of a image, shape (num_gts, h, w).
            gt_semantic_seg (list[tensor] | None): Each element is the ground
                truth of semantic segmentation with the shape (N, H, W).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            gt_caption_ids (list[Tensor]): Each element is the caption token ids.
            gt_caption_mask (list[Tensor]): Each element is the caption mask.
                1 represents valid token.
            gt_caption_nouns_ids (list[Tensor])
            gt_caption_nouns_mask (list[Tensor]): The two are same as above, except
                only object nouns are extracted.
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored. Defaults to None.
            kwargs:
                gt_cat_names (list[list[str]]): List of List of category names
                of the corresponding label in gt_labels

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # not consider ignoring bboxes
        assert gt_bboxes_ignore is None

        # forward
        all_cls_scores, all_cls_emb_preds, all_mask_preds = self(feats, img_metas)

        # preprocess ground truth
        gt_labels, gt_masks = self.preprocess_gt(gt_labels, gt_masks,
                                                 gt_semantic_seg, img_metas)

        # extract caption words' embeddings
        gt_caption_embs = None
        gt_caption_nouns_embs = None
        if self.use_caption_generation:
            gt_caption_embs, gt_caption_mask = \
                self.extract_word_embeddings(gt_caption_ids, gt_caption_mask, self.caption_gen_emb_type)
        if self.use_caption:
            gt_caption_nouns_embs, gt_caption_nouns_mask = \
                self.extract_word_embeddings(gt_caption_nouns_ids, gt_caption_nouns_mask, self.caption_emb_type)

        # loss
        losses = self.loss(all_cls_scores, all_cls_emb_preds, all_mask_preds,
                           gt_labels, gt_masks, gt_caption_ids, gt_caption_embs, gt_caption_mask,
                           gt_caption_nouns_ids, gt_caption_nouns_embs, gt_caption_nouns_mask, img_metas)

        return losses

    def simple_test(self, feats, img_metas, **kwargs):
        """Test without augmentaton.

        Args:
            feats (list[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two tensors.

            - mask_cls_results (Tensor): Mask classification logits,\
                shape (batch_size, num_queries, cls_out_channels).
                Note `cls_out_channels` should includes background.
            - mask_cls_emb_results (Tensor): embedding predictions,
                shape (batch_size, num_queries, d_l).
            - mask_pred_results (Tensor): Mask logits, shape \
                (batch_size, num_queries, h, w).
        """
        all_cls_scores, all_cls_emb_preds, all_mask_preds = self(feats, img_metas)
        mask_cls_results = all_cls_scores[-1]
        mask_cls_emb_results = all_cls_emb_preds[-1]
        mask_pred_results = all_mask_preds[-1]
        assigned_labels = mask_cls_results

        # assign results
        if kwargs.get('gt_labels', None) is not None:
            cls_emb_logits = self._get_cls_emb_logits(mask_cls_emb_results)
            gt_masks = kwargs['gt_masks'][0][0].pad(img_metas[0]['pad_shape'][:2], pad_val=0).to_tensor(dtype=torch.long, device=cls_emb_logits.device)
            # (num_queries, )
            assigned_labels, label_weights, mask_targets, mask_weights, pos_inds, neg_inds = \
                self._get_target_single(mask_cls_results[0], cls_emb_logits[0], mask_pred_results[0], kwargs['gt_labels'][0][0], gt_masks, img_metas)
            
        # upsample masks
        img_shape = img_metas[0]['batch_input_shape']
        if kwargs.get('img_shape', None):
            img_shape = kwargs['img_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=False)

        # caption generation result
        eval_types = self.test_cfg.get('eval_types', [])
        with_caption = kwargs.get('with_caption', False) or ('cap_results' in eval_types)
        caption_generation_results = None
        if with_caption:
            caption_generation_results = beam_search(self, mask_cls_emb_results, BOS_TOKEN, EOS_TOKEN, max_len=35, beam_width=7, logging=kwargs.get('logging', False))

        with_att = kwargs.get('with_att', False)
        att = None
        if with_att:
            nouns_ids = kwargs['nouns_ids']
            nouns_embs = get_ids_embedding(self, nouns_ids)
            att = torch.matmul(mask_cls_emb_results[0,:,:], nouns_embs.t())

        return assigned_labels, mask_cls_emb_results, mask_pred_results, caption_generation_results, att
