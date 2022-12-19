import torch

from models.musebert.musebert_model import MuseBERT

class EditMuseBERT(torch.nn.Module):
    def __init__(self, device, pretrained_path='../pretrained/musebert.pt', n_edit_types=4, n_decoder_layers=1):
        super(EditMuseBERT, self).__init__()
        
        self.encoder = MuseBERT.init_model(loss_inds=(0, 1, 2, 3, 4, 5, 6), relation_vocab_sizes=(5, 5, 5, 5)).to(device)
        self.encoder.load_model(pretrained_path, device)
        print(f'MuseBERT encoder loaded from: {pretrained_path}')

        self.edit_head = torch.nn.Linear(128, n_edit_types)

        self.decoder = MuseBERT.init_model(loss_inds=(0, 1, 2, 3, 4, 5, 6), relation_vocab_sizes=(5, 5, 5, 5), N=n_decoder_layers).to(device)

    def encode(self, data_in, rel_mat, mask):
        x = self.encoder.encode( data_in, rel_mat, mask)
        x = self.edit_head(x)
        return x
        
    def forward(self):
        # implement me
        return
