import os
import torch
import tqdm


class GatherImgFeature:

    def __init__(self, src_feat_dir):
        self.src_feat_dir = src_feat_dir

    def gather(self, save_name):
        src_feat_data = self._get_feature_data()
        torch.save(src_feat_data, save_name)
        print(f'{save_name}  files:{len(src_feat_data)}')
        del src_feat_data

    def _get_feature_data(self):
        files = os.listdir(self.src_feat_dir)
        src_feat_data = {}
        for file in tqdm.tqdm(files):
            if file.endswith('.pth'):
                img_name = file.split('.')[0]
                file_name = os.path.join(self.src_feat_dir, file)
                src_feat_data[img_name] = torch.load(file_name)
            else:
                print('no process file:{}'.format(file))

        return src_feat_data


if __name__ == '__main__':
    dataset_root = '/home/datasets/mix_data/mixk/data/datasets/'
    data_dir = 'vqa2/grid_features/features'
    datasets = ['train2014', 'test2015', 'val2014', 'visualgenome']

    for dataset in datasets:
        save_name = '{}.pt'.format(dataset)
        feature_dir = os.path.join(dataset_root, data_dir, dataset)
        print(f'processing {feature_dir}')

        ga = GatherImgFeature(src_feat_dir=feature_dir)
        ga.gather(save_name)
