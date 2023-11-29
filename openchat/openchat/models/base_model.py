from abc import ABC, abstractmethod
import cv2
import torch
import json
import logging
from ..utils import build_transforms, extract_visual_feature_on_single_image as extract_visual_feature
import os
import sys

sys.path.insert(0, '/home/datasets/mix_data/openchat/scene_graph_benchmark-main')


class BaseModel(ABC):

    def __init__(self, name: str, env):
        """
        Args:
            name (str): model name
            env (BaseEnv): dialogue manager
        """
        self.name = name
        self.env = env

    @abstractmethod
    def predict(self, user_id: str, text: str) -> str:
        """Predict output from histories and input text.

        Args:
            user_id (str): user's ID
            text (str): user's input text
        """

        return NotImplemented

    def run(self):
        self.env.run(self)


class MultiModel(ABC):

    def __init__(self, scene_graph_weight, cfg):
        self.device = 'cuda'
        self.mulitmodal_model = self.load_mulitmodal_model(cfg.model_weight)
        self.answer_table = self.load_answer_table(cfg.answer_table)
        self.scene_graph_model = self.load_scene_graph_model(scene_graph_weight)
        self.img_transforms = build_transforms()
        self.tokenizer = self.init_token(cfg.token) if hasattr(cfg, 'token') else None
        self.logger = logging.getLogger(__name__)

    def load_mulitmodal_model(self, model_path: str):
        model_path = os.path.normpath(model_path)
        model = torch.load(model_path).eval()
        return model

    def load_scene_graph_model(self, scene_graph_weight: str):
        scene_graph_weight = os.path.normpath(scene_graph_weight)
        model = torch.load(scene_graph_weight).eval()
        return model

    def load_answer_table(self, answer_table: str):
        answer_table = os.path.normpath(answer_table)
        file_format = answer_table.split('.')[-1]
        if file_format == 'json':
            with open(answer_table) as f:
                return json.load(f)

    def run_scene_graph_model(self, img_path: str):
        img = cv2.imread(img_path)
        return extract_visual_feature(self.scene_graph_model, self.img_transforms, img)

    def init_token(self, token_cfg):
        pass
