from typing import Union
from .envs import BaseEnv, TerminalEnv
import torch
from .models.imagemodel import LxmertBot, VilbertBot, OscarBot, VinVLBot, UNITERBot


class OpenChat(object):

    def __init__(
            self,
            model: str,
            device: str = 'cpu',
            env: Union[BaseEnv, str] = TerminalEnv(),
            max_context_length=128,
    ) -> None:
        """Constructor for OpenChat.

        Args:
            env (Union[BaseEnv, str]): dialogue environment
            model (str): generative dialogue model
            size (str): model size (It may vary depending on the model)
            device (str): device argument
            max_context_length (int): max history context length
                (it means that length of input context tokens)
        """

        self.device = device
        self.max_context_length = max_context_length
        self.env = env

        self.model = self.select_model(model)
        self.model.run()

    def select_model(self, model):
        assert model in self.available_models(), \
            f'Unsupported model. available models: {self.available_models()}'

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)

        model_map = {
            'vqa_model_lxmert': LxmertBot,
            'vqa_model_vilbert': VilbertBot,
            'vqa_model_oscar': OscarBot,
            'vqa_model_vinvl': VinVLBot,
            'vqa_model_uniter': UNITERBot,
        }

        vqa_model = model_map[model](
            env=self.env,
            max_context_length=self.max_context_length,
            device=self.device,
        )

        return vqa_model

    def available_models(self):
        return [
            'dialogpt',
            'vqa_model_lxmert',
            'vqa_model_vilbert',
            'vqa_model_oscar',
            'vqa_model_vinvl',
            'vqa_model_uniter',
        ]
