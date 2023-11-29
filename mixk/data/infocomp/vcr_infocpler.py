import numpy as np

from ..utils.stream import ItemFeature
from .base_infocpler import BaseInfoCpler


class VCRInfoCpler(BaseInfoCpler):

    def __init__(self, cfg):
        self.default_max_length = cfg.default_max_length
        self.max_ques_length = cfg.max_ques_length
        self.max_answer_length = cfg.max_answer_length
        self.max_obj_length = cfg.max_obj_length

    def complete_info(self, item_feature: ItemFeature):
        questions_embeddings, questions_masks = self.get_pad_ndarray([s.embs for s in item_feature.question.field_list],
                                                                     length=self.max_ques_length,
                                                                     need_mask=True,
                                                                     lowers=False)
        questions_tokens = [list(map(str, s.tokens)) for s in item_feature.question.field_list
                            ]  # [s.tokens for s in item_feature.question.field_list]
        questions_obj_tags = self.get_pad_2dlst([s.labels for s in item_feature.question_tags.field_list],
                                                length=self.max_ques_length,
                                                dtype=np.int,
                                                lowers=True)
        answers_embeddings, answers_masks = self.get_pad_ndarray([s.embs for s in item_feature.answers.field_list],
                                                                 length=self.max_answer_length,
                                                                 need_mask=True,
                                                                 lowers=False)
        answers_tokens = [list(map(str, s.tokens)) for s in item_feature.answers.field_list
                          ]  # [s.tokens for s in item_feature.answers.field_list]
        answers_obj_tags = self.get_pad_2dlst([s.labels for s in item_feature.answer_tags.field_list],
                                              length=self.max_answer_length,
                                              dtype=np.int,
                                              lowers=True)
        label = item_feature.label.label
        meta_data = item_feature.metadata
        segms = self.get_pad_segm(item_feature.segms.array, length=self.max_obj_length, lowers=True)
        objects = self.get_pad_1darray([s.label for s in item_feature.objects.field_list],
                                       length=self.max_obj_length,
                                       dtype=np.int,
                                       lowers=True)
        boxes = self.get_pad_2darray(
            item_feature.boxes.array, length=self.max_obj_length, dtype=np.float64, lowers=False)
        image = item_feature.image
        question_max_num = np.array([len(s.embs) for s in item_feature.question.field_list]).max()
        bbox_num = item_feature.boxes.array.shape[0]

        item_infos = {
            'questions_embeddings': questions_embeddings,
            'questions_masks': questions_masks,
            'questions_tokens': questions_tokens,
            'questions_obj_tags': questions_obj_tags,
            'answers_embeddings': answers_embeddings,
            'answers_masks': answers_masks,
            'answers_tokens': answers_tokens,
            'answers_obj_tags': answers_obj_tags,
            'label': label,
            'meta_data': meta_data,
            'segms': segms,
            'objects': objects,
            'boxes': boxes,
            'image': image,
            'question_max_num': question_max_num,
            'bbox_num': bbox_num,
        }
        return ItemFeature(item_infos)

    def get_pad_ndarray(self, lst, length=None, need_mask=False, lowers=False):
        if length is None:
            length = self.default_max_length

        spl = lst[0]
        n = len(lst)
        dim = spl.shape[-1]

        arr_pad = -2 * np.ones((n, length, dim), dtype=spl.dtype) if lowers else np.zeros(
            (n, length, dim), dtype=spl.dtype)
        for i, arr in enumerate(lst):
            lt = min([length, arr.shape[0]])
            arr_pad[i, :lt, :] = arr[:lt, :]
        if not need_mask:
            return arr_pad

        arr_mask = np.zeros((n, length), dtype=np.int)
        for i, arr in enumerate(lst):
            lt = min([length, arr.shape[0]])
            arr_mask[i, :lt] = np.ones(arr[:lt, 0].shape, dtype=np.int)
        return arr_pad, arr_mask

    def get_pad_1darray(self, lst, length=None, dtype=np.int, lowers=False):
        arr = np.array(lst, dtype=dtype)
        if length is None:
            length = self.default_max_length
        arr_pad = -2 * np.ones((length, ), dtype=dtype) if lowers else np.zeros((length, ), dtype=dtype)
        lt = min([length, arr.shape[0]])
        arr_pad[:lt] = arr[:lt]
        return arr_pad

    def get_pad_2darray(self, arr2d, length=None, dtype=None, lowers=False):
        dim = arr2d.shape[-1]
        if dtype is None:
            dtype = arr2d.dtype
        if length is None:
            length = self.default_max_length
        arr_pad = -2 * np.ones((length, dim), dtype=dtype) if lowers else -np.ones((length, dim), dtype=dtype)
        lt = min([length, arr2d.shape[0]])
        arr_pad[:lt, :] = arr2d[:lt, :]
        return arr_pad

    def get_pad_2dlst(self, lst, length=None, dtype=np.int, lowers=False):
        n = len(lst)
        if length is None:
            length = self.default_max_length
        arr_pad = -2 * np.ones((n, length), dtype=dtype) if lowers else np.zeros((n, length), dtype=dtype)
        for i, arr in enumerate(lst):
            lt = min([len(arr), length])
            arr_pad[i, :lt] = np.array(arr)[:lt]
        return arr_pad

    def get_pad_segm(self, segm, length=None, lowers=False):
        size_0, size_1 = segm.shape[1:]
        if length is None:
            length = self.default_max_length
        segm_pad = -2 * np.ones((length, size_0, size_1), dtype=segm.dtype) if lowers else np.zeros(
            (length, size_0, size_1), dtype=segm.dtype)
        lt = min([length, len(segm)])
        segm_pad[:lt] = segm[:lt]
        return segm_pad
