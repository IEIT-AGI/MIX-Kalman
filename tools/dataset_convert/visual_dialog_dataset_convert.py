import json
import numpy as np
import os
from shutil import copyfile


class VisDiaDatasetConvert:
    CONVERT_FUNC = {
        'train': 'decode_train_annotation',
        'val': 'decode_val_annotation',
        'test': 'decode_test_nnotation',
    }

    def __init__(self, json_files, save_dir):
        self._json_files = json_files
        self._save_dir = save_dir

    def convert(self):
        for file in self._json_files:
            print(file)
            if 'annotations' in file.split('/')[-1]:
                copyfile(file, self._save_dir)
            else:
                self.json2npy(file=file)

    def build_save_path(self, file):
        file_name = file.split('/')[-1]
        path = os.path.join(self._save_dir, file_name)
        return path.replace('.json', '.npy')

    def json2npy(self, file):

        json_data = self.read_json(file=file)

        func_key = ''
        for key in self.CONVERT_FUNC.keys():
            if key in file.split('/')[-1]:
                func_key = key
                break
        func = getattr(self, self.CONVERT_FUNC[func_key])
        annotation = func(json_data)

        self.save_npy_file(data=annotation, save_path=self.build_save_path(file=file))

    def decode_val_annotation(self, json_data):
        return self.decode_train_annotation(json_data)

    @staticmethod
    def decode_test_nnotation(json_data):
        data = json_data['data']
        split = json_data['split']
        version = json_data['version']

        questions = data['questions']
        answers = data['answers']
        dialogs = data['dialogs']

        vd_annotation = []  # visual_dialog_dataset
        vd_annotation.append({'split': split, 'version': version})

        def index_map_str(q_idx, a_idx, o_idxs: list = None):
            q = questions[q_idx]
            a = answers[a_idx] if a_idx else None
            os = list(answers[o_idx] for o_idx in o_idxs) if o_idxs else None
            return q, a, os

        vd_annotation = []  # visual_dialog_dataset
        vd_annotation.append({'split': split, 'version': version})
        for dl in dialogs:
            dialogs_data = {}
            dialogs_data['image_id'] = dl['image_id']
            dialogs_data['caption'] = dl['caption']
            dialogs_data['dialog'] = list()
            dialogs_data['round_id'] = dl['round_id']
            rounds = min(dl['round_id'], len(dl['dialog']))

            for idx, utterance in enumerate(dl['dialog'], start=1):
                q_idx = utterance['question']
                if idx < rounds:
                    a_idx = utterance['answer']
                    q, a, _ = index_map_str(q_idx, a_idx)
                    single_dialog = {
                        'question': q,
                        'answer': a,
                    }
                else:
                    a_ops_idx = utterance['answer_options']
                    q, _, os = index_map_str(q_idx, None, a_ops_idx)
                    single_dialog = {'question': q, 'answer_options': os}
                dialogs_data['dialog'].append(single_dialog)

            vd_annotation.append(dialogs_data)

        return vd_annotation

    @staticmethod
    def decode_train_annotation(json_data):
        data = json_data['data']
        split = json_data['split']
        version = json_data['version']

        questions = data['questions']
        answers = data['answers']
        dialogs = data['dialogs']

        def index_map_str(q_idx, a_idx, o_idxs: list = None):
            q = questions[q_idx]
            a = answers[a_idx]
            os = list(answers[o_idx] for o_idx in o_idxs)
            return q, a, os

        vd_annotation = []  # visual_dialog_dataset
        vd_annotation.append({'split': split, 'version': version})
        for dl in dialogs:
            dialogs_data = {}
            dialogs_data['image_id'] = dl['image_id']
            dialogs_data['caption'] = dl['caption']
            dialogs_data['dialog'] = list()

            for utterance in dl['dialog']:
                question = utterance['question']
                answer = utterance['answer']
                gt_index = utterance['gt_index']
                answer_options = utterance['answer_options']
                q, a, os = index_map_str(question, answer, answer_options)
                single_dialog = {'question': q, 'answer': a, 'gt_index': gt_index, 'answer_options': os}
                dialogs_data['dialog'].append(single_dialog)

            vd_annotation.append(dialogs_data)

        return vd_annotation

    @staticmethod
    def read_json(file):
        with open(file) as f:
            return json.load(f)

    @staticmethod
    def save_npy_file(data, save_path):
        np.save(save_path, data)


def build_small_dataset(size=128):
    dir = '/home/datasets/mix_data/mixk/data/datasets/visdial_data/annotations_npy'
    all_files = os.listdir(dir)
    files = [file for file in all_files if file.find('test') == -1]
    for file in files:
        file_name = os.path.join(dir, file)
        if file_name.endswith('.npy'):
            if file.find('val') != -1:
                size = int(size / 2)
            data = np.load(file_name, allow_pickle=True)
            small_file_name = os.path.join(file_name.split('.npy')[0] + f'_small{size}.npy')
            small_data = data[:size + 1]
            np.save(small_file_name, small_data)


def data_convert():
    data_root = '/home/datasets/mix_data/mixk/data/datasets/visdial_data/'
    json_path = os.path.join(data_root, 'annotations')
    save_path = os.path.join(data_root, 'annotations_npy_1')
    json_files = os.listdir(json_path)
    json_files = list(os.path.join(json_path, json_file) for json_file in json_files)
    vdc = VisDiaDatasetConvert(json_files=json_files, save_dir=save_path)
    vdc.convert()


def convert_train_dense():
    data_root = '/home/datasets/mix_data/mixk/'
    annotation_path = 'data/datasets/visdial_data/annotations_npy/'
    json_file = os.path.join(data_root, annotation_path, 'visdial_1.0_train_dense_processed.json')

    # json_dense = os.path.join(data_root, annotation_path, 'visdial_1.0_train_dense_annotations_processed.json')
    # j = json.load(open(json_dense, 'r'))

    json_data = VisDiaDatasetConvert.read_json(file=json_file)
    data = VisDiaDatasetConvert.decode_train_annotation(json_data)
    save_name = json_file.replace('.json', '.npy')
    VisDiaDatasetConvert.save_npy_file(data, save_path=save_name)


if __name__ == '__main__':
    # data_convert()
    # build_small_dataset()
    convert_train_dense()
