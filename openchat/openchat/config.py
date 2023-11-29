from mixk.utils.config import mixkEasyDict

dataset_root = '/home/datasets/mix_data/'
openchat_path = dataset_root + 'openchat/'
image_path = openchat_path + 'static/image/'

detect_weight_path = openchat_path + 'model_pth/detect.pth'

# lxmert_weight_path = os.path.join(root_path, 'model_pth/lxmert_vqa.pth')
# vilbert_weight_path = os.path.join(root_path, 'model_pth/vilbert_vqa.pth')
# oscar_weight_path = os.path.join(root_path, 'model_pth/oscar_vqa.pth')
# vinvl_weight_path = os.path.join(root_path, 'model_pth/vinvl_vqa.pth')
# devlbert_weight_path = os.path.join(root_path, 'model_pth/devlbert_vqa.pth')
# uniter_weight_path = os.path.join(root_path, 'model_pth/uniter_vqa.pth')

# answer_table = json.load(open('/home/datasets/mix_data/lxmert/vqa/trainval_label2ans.json'))

model_root_path = openchat_path + '/model_pth/'
model_vqa_path = dict(
    lxmert=dict(
        model_weight=model_root_path + 'lxmert_vqa.pth',
        answer_table=dataset_root + 'lxmert/vqa/trainval_label2ans.json'),
    vilbert=dict(
        model_weight=model_root_path + 'vilbert_vqa.pth',
        answer_table=dataset_root + 'vilbert/datasets/VQA/cache/trainval_label2ans.json',
        token=dict(pretrained_model_name_or_path='bert-base-uncased', do_lower_case=True)),
    oscar=dict(
        model_weight=model_root_path + 'oscar_vqa.pth',
        answer_table=dataset_root + 'vilbert/datasets/VQA/cache/trainval_label2ans.json',
        token=dict(
            pretrained_model_name_or_path=dataset_root + 'model/oscar/base-vg-labels/ep_107_1192087',
            do_lower_case=True)),
    uniter=dict(
        model_weight=model_root_path + 'uniter_vqa.pth',
        answer_table=dataset_root + 'vilbert/datasets/VQA/cache/trainval_label2ans.json',
        token=dict(pretrained_model_name_or_path='bert-base-uncased', do_lower_case=True)),
    vinvl=dict(
        model_weight=model_root_path + 'vinvl_vqa.pth',
        answer_table=dataset_root + 'vilbert/datasets/VQA/cache/trainval_label2ans.json',
        token=dict(
            pretrained_model_name_or_path=dataset_root + 'model/oscar/base-vg-labels/ep_107_1192087',
            do_lower_case=True)),
)

model_vqa_path = mixkEasyDict(model_vqa_path)
