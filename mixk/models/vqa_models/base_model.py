from abc import ABCMeta

import torch.nn as nn
# class BaseModel(nn.Module):
#
#   def __init__(self):
#     super().__init__()
#
#   def forward(self, data, **kwargs):
#     if self.training:
#       losses = self.forward_train(data, **kwargs)
#       loss, losses_log = self._parse_losses(losses=losses)
#       model_outputs = dict()
#       model_outputs.update(losses_log)
#       model_outputs['loss'] = loss
#       return model_outputs
#     else:
#       return self.forward_test(data, **kwargs)
#
#   def forward_train(self, data, **kwargs):
#     pass
#
#   def forward_test(self, data, **kwargs):
#     pass
#
#   @staticmethod
#   def _parse_losses(losses: Dict) -> Tuple[torch.Tensor, Dict]:
#
#     losses_log = OrderedDict()
#     for name, value in losses.items():
#       losses_log[name] = value.mean()
#
#     loss = sum(v for k, v in losses_log.items() if 'loss' in k)
#     losses_log['loss'] = loss
#     for name, value in losses_log.items():
#       if dist.is_available() and dist.is_initialized():
#         loss_value = value.data.clone()
#         value = loss_value.div_(dist.get_world_size())
#         dist.all_reduce(value)
#
#       losses_log[name] = value.item()
#     return loss, losses_log


class BaseModel(nn.Module, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

    # def forward(self, data, **kwargs):
    #     if self.training:
    #         losses = self.forward_train(data, **kwargs)
    #         loss, losses_log = self._parse_losses(losses=losses)
    #         model_outputs = dict()
    #         model_outputs.update(losses_log)
    #         model_outputs['loss'] = loss
    #         return model_outputs
    #     else:
    #         return self.forward_test(data, **kwargs)

    def forward(self, data, **kwargs):
        if self.training:
            return self.forward_train(data, **kwargs)
        else:
            return self.forward_test(data, **kwargs)

    def forward_train(self, data, **kwargs):
        pass

    def forward_test(self, data, **kwargs):
        pass
