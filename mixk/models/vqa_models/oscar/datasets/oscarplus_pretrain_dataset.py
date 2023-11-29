from .oscar_tsv import OscarTSVDataset
from mixk.data.builder import DATASETS
from pytorch_transformers import BertTokenizer
from torch.utils.data import Dataset


@DATASETS.register_module()
class OscarPretrainDataset(Dataset):

    def __init__(self, reader):
        super(OscarPretrainDataset, self).__init__()
        token_cfg = reader.token
        oscar_tsv_dataset_cfg = reader.oscar_tsv_dataset
        extra_dataset_cfg = reader.extra_dataset

        self.token_cfg = token_cfg
        self.oscar_tsv_dataset_cfg = oscar_tsv_dataset_cfg
        self.extra_dataset_cfg = extra_dataset_cfg

        tokenizer = self.build_tokenizer()
        self.datasets = list(self.make_dataset(tokenizer))

        if extra_dataset_cfg:
            self.datasets.append(self.make_extra_dataset(tokenizer))

    def build_tokenizer(self):
        return BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.token_cfg.name, do_lower_case=self.token_cfg.do_lower_case)

    def make_dataset(self, tokenizer):
        data_cfg = self.oscar_tsv_dataset_cfg
        cfg = dict(
            yaml_file=data_cfg.dataset_yaml_file,
            args=data_cfg.args,
            seq_len=data_cfg.args.max_seq_length,
            on_memory=data_cfg.on_memory,
            tokenizer=tokenizer)

        return OscarTSVDataset(**cfg)

    def make_extra_dataset(self, tokenizer):
        data_cfg = self.oscar_tsv_dataset_cfg
        cfg = dict(
            yaml_file=self.extra_dataset_cfg.dataset_yaml_file,
            args=data_cfg.args,
            seq_len=data_cfg.args.max_seq_length,
            on_memory=data_cfg.on_memory,
            tokenizer=tokenizer,
            textb_sample_mode=self.extra_dataset_cfg.extra_textb_sample_mode)
        return OscarTSVDataset(**cfg)

    def __getitem__(self, item):
        return self.datasets[item]

    def __len__(self):
        return self.datasets.__len__()
