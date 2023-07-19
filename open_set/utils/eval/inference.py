# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

import transformers

from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter

from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

def inference_detector(model, imgs, **kwargs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.
        with_caption (bool): Whether inference caption generation results.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        results = model(data['img'], data['img_metas'], return_loss=False, rescale=True, **kwargs)

    return results

def get_ids_embedding(model, ids):
    emb = model.bert_embeddings.word_embeddings(ids)
    emb = model.bert_embeddings.LayerNorm(emb).squeeze(0)
    # if len(emb.shape) == 2:
        # emb = emb.unsqueeze(0)
    return emb

def beam_search(model, memory, BOS, EOS, max_len, beam_width=7, alpha=0.7, logging=False):
    device = memory.device
    target = torch.tensor([[[BOS]]])
    target_emb = get_ids_embedding(model, target.to(device))
    outputs = model.caption_generator(
        tgt=target_emb.to(device),
        memory=memory)[0]
    outputs = torch.stack([model.caption_generator.generator(output[0, 0, :]) for output in outputs], dim=0)
    logits = torch.mean(outputs, dim=0)

    scaled_logits = torch.log_softmax(logits[None, :], dim=1).cpu().squeeze(0) # over vocab size 
    weights, candidates = torch.topk(input=scaled_logits, k=beam_width, largest=True)
    
    response_tracker = []  # for valid final sequence 
    sequence_tracker = []  # for current active sequence
    for idx in candidates:
        option = torch.tensor([[idx]])  # a new option into the search tree 
        sequence = torch.cat([target.squeeze(0), option], dim=1)
        sequence_tracker.append(sequence)
    
    keep_generating = True
    while keep_generating:
        input_batch = torch.vstack(sequence_tracker)
        input_batch = get_ids_embedding(model, input_batch.to(device))
        with torch.no_grad():
            input_memory = torch.cat([m.repeat(input_batch.shape[0], 1, 1) for m in memory], dim=0)
            outputs = model.caption_generator(
                input_batch.to(device),
                input_memory)[0]
            logits = torch.mean(torch.stack([model.caption_generator.generator(out[:, -1, :]) for out in outputs]), dim=0)
        scaled_logits = torch.log_softmax(logits, dim=1).cpu()
        
        length = input_batch.shape[1]
        vocab_size = scaled_logits.shape[1]
        weighted_logits = (scaled_logits + weights[:, None]) / length ** alpha  
        weights, candidates = torch.topk(torch.flatten(weighted_logits), k=beam_width, largest=True)
        weights = weights * length ** alpha  # denormalize

        weights_tmp = []
        sequence_tmp = []
        max_score = -100
        max_idx = 0
        for idx, pos in enumerate(candidates):
            row = torch.div(pos, vocab_size, rounding_mode='floor') # get relative position over nb_sequences 
            col = pos % vocab_size  # get relative position over vocab_size 
            sequence = torch.cat([sequence_tracker[row], torch.tensor([[col]])], dim=1)
            if col == EOS:
                flattened_sequence = torch.flatten(sequence).tolist()
                sequence_score = weights[idx] / len(flattened_sequence) ** alpha 
                response_tracker.append((flattened_sequence, sequence_score))  # a sentence was built 
                if sequence_score > max_score:
                    max_score = sequence_score
                    max_idx = len(response_tracker) - 1
                if len(response_tracker) == beam_width:
                    keep_generating = False 
                    break  # end the for loop over candidates
            elif sequence.shape[1] < max_len - 1:
                weights_tmp.append(weights[row])
                sequence_tmp.append(sequence)
        # end for loop over candidates ...!

        if len(sequence_tmp) == 0: 
            keep_generating = False 
        else:               
            weights = torch.tensor(weights_tmp)
            sequence_tracker = sequence_tmp
    # end while search loop ...!
    bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    res_sentence = ''
    for i, (sentence, score) in enumerate(response_tracker):
        sentence = bert_tokenizer.decode(sentence)
        if i == max_idx:
            res_sentence = sentence[1:-1]
        if logging:
            print(sentence, score)
    return res_sentence