import sys
sys.path.append('.')

from torch import optim, nn, utils, Tensor
from . import sashimi.Sashimi as Sashimi
from transformers import GPTModel, DistilBertModel

class SparseAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_RNN = Sashimi()
        self.soft_masker = nn.Softmax(dim=1)
        self.decoder_transformer = DistilBertModel.from_pretrained('distilbert-base-uncased')

    def forward(self, x):
        #x: B,L,D
        soft_mask = self.encoderRNN(x)
        top_values, top_indices = torch.topk(soft_mask, k=500, dim=1)
        selected_values = torch.gather(x, dim=1, index=top_indices)
        out = self.decoderGPT(selected_values)


        return out
