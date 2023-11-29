import os
import time
import numpy as np
import torch
from ..utils.task_utils import (convert_examples_to_features_vqa, output_modes, processors)
import sys

sys.path.insert(0, '.')


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    processor = processors[task]()
    output_mode = output_modes[task]

    label_list = processor.get_labels(args.label_file)

    t_start = time.time()
    examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)

    # img_features = torch.load(os.path.join(args.data_dir, 'val_img_feats.pt' if evaluate else 'train_img_feats.pt'))
    # img_features = torch.load(os.path.join(args.data_dir, 'val_img_frcnn_feats.pt' \
    # if evaluate else 'train_img_frcnn_feats.pt'))
    img_features = np.load(
        os.path.join(args.data_dir, 'val_img_frcnn_feats.npy' if evaluate else 'train_img_frcnn_feats.npy'))

    features = convert_examples_to_features_vqa(
        examples,
        img_features,
        label_list,
        args.max_img_seq_length,
        args.max_seq_length,
        tokenizer,
        output_mode,
        cls_token_at_end=bool(args.model_type in ['xlnet']),  # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        sep_token=tokenizer.sep_token,
        cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
        pad_on_left=bool(args.model_type in ['xlnet']),  # pad on the left for xlnet
        pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)

    # if args.local_rank in [-1, 0]:
    #     logger.info("Saving features into cached file %s", cached_features_file)
    #     torch.save(features, cached_features_file)
    t_end = time.time()
    logger.info('Info: loading features using %.5f secs' % (t_end - t_start))

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)  # batch*max_seq_len
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == 'classification':
        labels = torch.tensor([f.label_id[0] for f in features], dtype=torch.long)
        targets = torch.tensor([target_tensor(len(label_list), f.label_id, f.score) for f in features],
                               dtype=torch.float)

        if args.img_feature_dim > 0:  # change here
            t_start = time.time()
            img_feat_np = np.zeros((labels.shape[0], args.max_img_seq_length, args.img_feature_dim))
            for f_id, f in enumerate(features):
                img_feat_np[f_id] = f.img_feat

            img_feats = torch.from_numpy(img_feat_np)

            # img_feats = torch.empty((labels.shape[0], args.max_img_seq_length, args.img_feature_dim))
            # for f_id, f in enumerate(features):
            #    img_feats[f_id] = f.img_feat

            t_end = time.time()
            logger.info('Info: convert image tensor features using %.5f secs' % (t_end - t_start))

            # img_feats = torch.stack([f.img_feat[:,-args.img_feature_dim:] for f in features])
            # img_feats = torch.stack([f.img_feat for f in features])
        # img_feats = img_feats.type(torch.long)

        # print('targets:', targets.shape)
        print('img_feats:', img_feats.shape)
    elif output_mode == 'regression':
        # all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
        pass

    if args.img_feature_dim == -1:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, labels, targets)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, labels, targets, img_feats)
    return dataset


def target_tensor(len, labels, scores):
    """create the target by labels and scores."""
    target = [0] * len
    for id, l in enumerate(labels):
        target[l] = scores[id]

    return target
