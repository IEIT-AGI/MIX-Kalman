import os
import torch


class VisDiaBertModelConvert:

    def __init__(self, model_dir: str, save_dir: str):
        self.model_dir = model_dir
        self.save_dir = save_dir

    def convert(self):
        for model_path in os.listdir(self.model_dir):
            save_path = self.build_save_path(model_path)
            model_path = os.path.join(self.model_dir, model_path)
            model = torch.load(model_path, map_location='cpu')
            self.save_model(model_data=model, path=save_path)
            print('{} convert to {}'.format(model_path, save_path))

    def build_save_path(self, file_name):
        return os.path.join(self.save_dir, file_name + '.pth')

    @staticmethod
    def save_model(model_data, path):
        if 'vqa_weights' in path:
            data = {'model': model_data}
        else:
            data = {'model': model_data['model_state_dict']}
        with open(path, 'wb') as fwb:
            torch.save(data, fwb)


if __name__ == '__main__':
    source_model_dir = '/home/datasets/mix_data/model/visdial'
    save_dir = '/home/datasets/mix_data/model/visdial_model_mixk'
    vdbmc = VisDiaBertModelConvert(model_dir=source_model_dir, save_dir=save_dir)
    vdbmc.convert()
