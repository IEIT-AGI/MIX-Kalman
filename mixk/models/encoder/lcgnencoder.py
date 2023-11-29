import numpy as np
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from ..builder import ENCODER
import torch.nn as nn
import torch


@ENCODER.register_module()
class LCGNEncoder(nn.Module):

    def __init__(self, WRD_EMB_INIT_FILE: str, encInputDropout: float, qDropout: float, WRD_EMB_DIM: int, ENC_DIM: int,
                 WRD_EMB_FIXED: bool) -> None:
        """Initialization of LCGNEncoder.

        Args:
          WRD_EMB_INIT_FILE: the file path storing the initial information of word embedding
          encInputDropout: dropout rate of encoder input
          qDropout: question dropout
          WRD_EMB_DIM: the dimension of word embedding
          ENC_DIM: the dimension of encoder
          WRD_EMB_FIXED: if the word embedding is fixed during training
        """
        super().__init__()
        self.WRD_EMB_INIT_FILE = WRD_EMB_INIT_FILE
        self.encInputDropout = encInputDropout
        self.qDropout = qDropout
        self.WRD_EMB_DIM = WRD_EMB_DIM
        self.ENC_DIM = ENC_DIM
        self.WRD_EMB_FIXED = WRD_EMB_FIXED
        embInit = np.load(self.WRD_EMB_INIT_FILE)  # shape: (2956,300)
        self.embeddingsVar = nn.Parameter(torch.Tensor(embInit), requires_grad=(not self.WRD_EMB_FIXED)).cuda()
        self.enc_input_drop = nn.Dropout(1 - self.encInputDropout)
        self.rnn0 = BiLSTM(self.WRD_EMB_DIM, self.ENC_DIM)
        self.question_drop = nn.Dropout(1 - self.qDropout)

    def forward(self, qIndices: Tensor, questionLengths: Tensor) -> Tuple[Tensor, Tensor]:
        """forward computatuion of LCGNEncoder, based on inputs.
        Args:
          qIndices: the indices of questions, shape of batch_size x 128
          questionLengths: the length of the question, shape of batch_size

        Returns:
          Tuple[Tensor, Tensor]: questionCntxWords: the representation of word context in questions, shape of
            batch_size x128x512;
          vecQuestions: the representation of the whole question, shape of batch_size x512
        """
        # Word embedding
        # embeddingsVar = self.embeddingsVar.cuda()
        # embeddings = torch.cat(
        #     [torch.zeros(1, self.WRD_EMB_DIM, device='cuda'), embeddingsVar],
        #     dim=0)
        embeddingsVar = self.embeddingsVar
        embeddings = torch.cat([torch.zeros(1, self.WRD_EMB_DIM, device='cpu').cuda(), embeddingsVar], dim=0)

        questions = F.embedding(qIndices, embeddings)
        questions = self.enc_input_drop(questions)

        # RNN (LSTM)
        questionCntxWords, vecQuestions = self.rnn0(questions, questionLengths)
        vecQuestions = self.question_drop(vecQuestions)

        return questionCntxWords, vecQuestions


class BiLSTM(nn.Module):

    def __init__(self, WRD_EMB_DIM: int, ENC_DIM: int, forget_gate_bias: float = 1.) -> None:
        """Initialization of BiLSTM.

        Args:
          WRD_EMB_DIM: the word embedding dimension
          ENC_DIM: the dimension of the encoder for BiLSTM, which is twice of the hidden state dimension
          forget_gate_bias:(optional):the initialization of the forget-gate bias
        """
        super().__init__()
        self.WRD_EMB_DIM = WRD_EMB_DIM
        self.ENC_DIM = ENC_DIM

        self.bilstm = torch.nn.LSTM(
            input_size=self.WRD_EMB_DIM,
            hidden_size=self.ENC_DIM // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True)

        d = self.ENC_DIM // 2

        # initialize LSTM weights (to be consistent with TensorFlow)
        fan_avg = (d * 4 + (d + self.WRD_EMB_DIM)) / 2.
        bound = np.sqrt(3. / fan_avg)
        nn.init.uniform_(self.bilstm.weight_ih_l0, -bound, bound)
        nn.init.uniform_(self.bilstm.weight_hh_l0, -bound, bound)
        nn.init.uniform_(self.bilstm.weight_ih_l0_reverse, -bound, bound)
        nn.init.uniform_(self.bilstm.weight_hh_l0_reverse, -bound, bound)

        # initialize LSTM forget gate bias (to be consistent with TensorFlow)
        self.bilstm.bias_ih_l0.data[...] = 0.
        self.bilstm.bias_ih_l0.data[d:2 * d] = forget_gate_bias
        self.bilstm.bias_hh_l0.data[...] = 0.
        self.bilstm.bias_hh_l0.requires_grad = False
        self.bilstm.bias_ih_l0_reverse.data[...] = 0.
        self.bilstm.bias_ih_l0_reverse.data[d:2 * d] = forget_gate_bias
        self.bilstm.bias_hh_l0_reverse.data[...] = 0.
        self.bilstm.bias_hh_l0_reverse.requires_grad = False

    def forward(self, questions: Tensor, questionLengths: Tensor) -> Tuple[Tensor, Tensor]:
        """encoder based on BiLSTM.

        Args:
          questions: the question representation, sized of 128x128x300
          questionLengths: the question length which is 128, recoding the length of questions.

        Returns:
          Tuple[Tensor, Tensor]: output:the output of bilstm, sized of batch_size x128x512; h_n:
            the hidden states of bilstm, sized of batch_size x 512
        """
        # sort samples according to question length (descending)
        sorted_lengths, indices = torch.sort(questionLengths, descending=True)
        sorted_questions = questions[indices]
        _, desorted_indices = torch.sort(indices, descending=False)

        # pack questions for LSTM forwarding
        sorted_lengths = sorted_lengths.to(device='cpu', dtype=torch.int64)
        packed_questions = nn.utils.rnn.pack_padded_sequence(sorted_questions, sorted_lengths, batch_first=True)
        packed_output, (sorted_h_n, _) = self.bilstm(packed_questions)
        sorted_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=questions.size(1))
        sorted_h_n = torch.transpose(sorted_h_n, 1, 0).reshape(questions.size(0), -1)

        # sort back to the original sample order
        output = sorted_output[desorted_indices]
        h_n = sorted_h_n[desorted_indices]

        return output, h_n
