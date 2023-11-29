import copy
import json
import os

import h5py
import numpy as np
from allennlp.data.fields import ArrayField, LabelField, ListField, MetadataField
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.data.vocabulary import Vocabulary

from ..utils.image_utils import load_image, resize_image, to_tensor_and_normalize
from ..utils.stream import ItemFeature
from ..utils.vcr_utils import _fix_tokenization, make_mask
from .base_reader import mixkDataReader


class VCRReader(mixkDataReader):

    def __init__(self, cfg):
        splits = cfg.datasets
        if isinstance(splits, str):
            splits = [splits]
        self.mode = cfg.mode
        assert self.mode in ['answer', 'rationale']
        self.only_use_relevant_dets = cfg.get('only_use_relevant_dets', True)
        self.add_image_as_a_box = cfg.get('add_image_as_a_box', True)
        self.is_train = cfg.get('is_train', True)
        self.token_indexers = {'elmo': ELMoTokenCharactersIndexer()}

        with open(os.path.join(cfg['coco_cate_dir']), 'r') as f:
            coco_cate = json.load(f)
        self.coco_objects = ['__background__'
                             ] + [x['name'] for k, x in sorted(coco_cate.items(), key=lambda x: int(x[0]))]
        self.coco_obj_to_ind = {o: i for i, o in enumerate(self.coco_objects)}

        self.vocab = Vocabulary()

        self.img_dir = cfg.image_dir
        self.annotation_pathes = {split: cfg['annotations'][split] for split in splits}
        self.grp_items_dict = {
            'answer': {split: cfg['text_infos']['answer'][split]
                       for split in splits},
            'rationale': {split: cfg['text_infos']['rationale'][split]
                          for split in splits}
        }

        self.annotations = []
        self.annotations_idx_split = []
        for split, annotation_path in self.annotation_pathes.items():
            annotations_now = self._load_jsonl(annotation_path)
            self.annotations.extend(annotations_now)
            self.annotations_idx_split.extend(copy.copy([split] * len(annotations_now)))

        self.default_answer_idx = cfg.default_answer_idx

    def _load_jsonl(self, annotation_path):
        with open(annotation_path, 'r') as f:
            vs = [json.loads(s) for s in f]
        return vs

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):

        annotation = self.annotations[item]
        split = self.annotations_idx_split[item]
        item_feature = ItemFeature(annotation)
        item_feature.error = False
        # for k, v in annotation.items():
        #    item_feature[k] = v

        if self.mode == 'rationale':
            conditioned_label = item_feature['answer_label'] if split != 'test' else self.default_answer_idx
            item_feature['question_only'] = copy.deepcopy(item_feature['question'])
            item_feature['question'] = item_feature['question'] + item_feature['answer_choices'][conditioned_label]
            # answer_conditioned = item_feature['answer_choices'][conditioned_label]

        answer_choices = item_feature['{}_choices'.format(self.mode)]
        dets2use, old_det_to_new_ind = self._get_dets_to_use(annotation)

        with h5py.File(self.grp_items_dict[self.mode][split], 'r') as h5:
            grp_info = {k: np.array(v, dtype=np.float16) for k, v in h5[str(item)].items()}

        condition_key = self.default_answer_idx if split == 'test' and self.mode == 'rationale' else ''
        instance_dict = {}
        questions_tokenized, question_tags = zip(*[
            _fix_tokenization(
                item_feature['question'],
                grp_info[f'ctx_{self.mode}{condition_key}{i}'],
                old_det_to_new_ind,
                item_feature['objects'],
                token_indexers=self.token_indexers,
                pad_ind=0 if self.add_image_as_a_box else -1) for i in range(4)
        ])
        instance_dict['question'] = ListField(questions_tokenized)
        instance_dict['question_tags'] = ListField(question_tags)

        answers_tokenized, answer_tags = zip(*[
            _fix_tokenization(
                answer,
                grp_info[f'answer_{self.mode}{condition_key}{i}'],
                old_det_to_new_ind,
                item_feature['objects'],
                token_indexers=self.token_indexers,
                pad_ind=0 if self.add_image_as_a_box else -1) for i, answer in enumerate(answer_choices)
        ])

        instance_dict['answers'] = ListField(answers_tokenized)
        instance_dict['answer_tags'] = ListField(answer_tags)
        if split != 'test':
            instance_dict['label'] = LabelField(item_feature['{}_label'.format(self.mode)], skip_indexing=True)
        instance_dict['metadata'] = MetadataField({
            'annot_id': item_feature['annot_id'],
            'ind': item,
            'movie': item_feature['movie'],
            'img_fn': item_feature['img_fn'],
            'question_number': item_feature['question_number']
        })

        image = load_image(os.path.join(self.img_dir, item_feature['img_fn']))
        image, window, img_scale, padding = resize_image(image, random_pad=self.is_train)
        image = to_tensor_and_normalize(image)
        c, h, w = image.shape

        with open(os.path.join(self.img_dir, item_feature['metadata_fn']), 'r') as f:
            metadata = json.load(f)
        segms = np.stack(
            [make_mask(mask_size=14, box=metadata['boxes'][i], polygons_list=metadata['segms'][i]) for i in dets2use])

        # Chop off the final dimension, that"s the confidence
        boxes = np.array(metadata['boxes'])[dets2use, :-1]
        # Possibly rescale them if necessary
        boxes *= img_scale
        boxes[:, :2] += np.array(padding[:2])[None]
        boxes[:, 2:] += np.array(padding[:2])[None]
        obj_labels = [self.coco_obj_to_ind[item_feature['objects'][i]] for i in dets2use.tolist()]
        if self.add_image_as_a_box:
            boxes = np.row_stack((window, boxes))
            segms = np.concatenate((np.ones((1, 14, 14), dtype=np.float32), segms), 0)
            obj_labels = [self.coco_obj_to_ind['__background__']] + obj_labels

        instance_dict['segms'] = ArrayField(segms, padding_value=0)
        instance_dict['objects'] = ListField([LabelField(x, skip_indexing=True) for x in obj_labels])

        if not np.all((boxes[:, 0] >= 0.) & (boxes[:, 0] < boxes[:, 2])):
            import ipdb
            ipdb.set_trace()
        assert np.all((boxes[:, 1] >= 0.) & (boxes[:, 1] < boxes[:, 3]))
        assert np.all((boxes[:, 2] <= w))
        assert np.all((boxes[:, 3] <= h))
        instance_dict['boxes'] = ArrayField(boxes, padding_value=-1)

        # ins = Instance(instance_dict)
        # ins.index_fields(self.vocab)

        instance_dict['image'] = image
        out = ItemFeature(instance_dict)
        return out

    def _get_dets_to_use(self, annotation):
        """We might want to use fewer detectiosn so lets do so.

        :param item:
        :param question:
        :param answer_choices:
        :return:
        """
        # Load questions and answers
        question = annotation['question']
        answer_choices = annotation['{}_choices'.format(self.mode)]

        if self.only_use_relevant_dets:
            dets2use = np.zeros(len(annotation['objects']), dtype=bool)
            people = np.array([x == 'person' for x in annotation['objects']], dtype=bool)
            for sent in answer_choices + [question]:
                for possibly_det_list in sent:
                    if isinstance(possibly_det_list, list):
                        for tag in possibly_det_list:
                            if tag >= 0 and tag < len(annotation['objects']):  # sanity check
                                dets2use[tag] = True
                    elif possibly_det_list.lower() in ('everyone', 'everyones'):
                        dets2use |= people
            if not dets2use.any():
                dets2use |= people
        else:
            dets2use = np.ones(len(annotation['objects']), dtype=bool)

        # we will use these detections
        dets2use = np.where(dets2use)[0]

        old_det_to_new_ind = np.zeros(len(annotation['objects']), dtype=np.int32) - 1
        old_det_to_new_ind[dets2use] = np.arange(dets2use.shape[0], dtype=np.int32)

        # If we add the image as an extra box then the 0th will be the image.
        if self.add_image_as_a_box:
            old_det_to_new_ind[dets2use] += 1
        old_det_to_new_ind = old_det_to_new_ind.tolist()
        return dets2use, old_det_to_new_ind
