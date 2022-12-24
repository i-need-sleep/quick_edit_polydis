import torch

from models.polydis.ptvae import RnnEncoder
from models.polydis.amc_dl.torch_plus.train_utils import get_zs_from_dists
from models.musebert.musebert_model import MuseBERT

class EditMuseBERT(torch.nn.Module):
    def __init__(self, device, pretrained_path='../pretrained/musebert.pt', n_edit_types=128, max_n_inserts=16, n_decoder_layers=2):
        super(EditMuseBERT, self).__init__()
        
        self.max_n_inserts = max_n_inserts
        
        # Load a pretrained MuseBERT encoder
        self.encoder = MuseBERT.init_model(loss_inds=(0, 1, 2, 3, 4, 5, 6), relation_vocab_sizes=(5, 5, 5, 5)).to(device)
        self.encoder.load_model(pretrained_path, device)
        print(f'MuseBERT encoder loaded from: {pretrained_path}')

        self.step_embs = torch.nn.Embedding(32, 128)
        self.edit_head = torch.nn.Linear(128, n_edit_types)
        self.n_inserts_head = torch.nn.Linear(128, max_n_inserts) # num of inserts at each onset step

        # Chord encoder from Polydis (from scratch)
        self.chord_enc = RnnEncoder(36, 1024, 128)

        # Decoder
        self.decoder = MuseBERT.init_model(loss_inds=(0, 1, 2, 3, 4, 5, 6), relation_vocab_sizes=(5, 5, 5, 5), N=n_decoder_layers).to(device)
        self.decoder_splits = [9, 7, 7, 3, 12, 5, 8] # Sizes for each dim of the factorised features
    
    def encode_chd(self, chd):
        x = self.chord_enc(chd)
        z_chd = get_zs_from_dists([x], False)[0]
        return z_chd

    def encode(self, editor_in, z_chd):
        [data_in, rel_mat_in, length] = editor_in

        # Embed the time step tokens (for predicting n_inserts)
        onset_steps = torch.tensor([[i for i in range(32)] for j in range(data_in.shape[0])])
        onset_embs = self.step_embs(onset_steps)

        # Embed the note atrs
        x = self.encoder.onset_pitch_dur_embedding(data_in)

        # Update the input embs: left to right: [z_chd, onset_embs, x]
        z_chd = z_chd.unsqueeze(1)
        x = torch.cat((z_chd, onset_embs, x), dim=1)

        # Build output masks
        n_inserts_mask = torch.zeros(x.shape[0], x.shape[1])
        n_inserts_mask[:, 1: 33] = 1
        edit_mask = torch.zeros(x.shape[0], x.shape[1])

        # Update the rel_mat and the mask
        rel_mat = torch.zeros(x.shape[0], 4, x.shape[1], x.shape[1])
        rel_mat[:, :, -100:, -100:] = rel_mat_in
        rel_mat = rel_mat.int()

        mask = []
        for idx, l in enumerate(length):
            mask_l = torch.zeros(x.shape[1], x.shape[1])
            mask_l[: 33+l, :33+l] = 1
            mask.append(mask_l)
            edit_mask[idx, 33: 33+l] = 1
        mask = torch.stack(mask, dim=0).int()

        # Forward pass through the transformer
        x = self.encoder.tfm(x, rel_mat, mask=mask)

        n_inserts_out = self.n_inserts_head(x)[n_inserts_mask > 0]
        edits_out = self.edit_head(x)[edit_mask > 0]
        z_pool = x[:, 0:1, :]
        
        return z_pool, edits_out, n_inserts_out

    def decode(self, decoder_in, z_chd, output_mask):

        [atr_in, rel_in, length] = decoder_in

        # Embed the note atrs
        x = self.decoder.onset_pitch_dur_embedding(atr_in)

        # Update the input embs: left to right: [z_chd, x]
        z_chd = z_chd.unsqueeze(1)
        x = torch.cat((z_chd, x), dim=1)

        # Update the rel_mat and the mask
        rel_mat = torch.zeros(x.shape[0], 4, x.shape[1], x.shape[1])
        rel_mat[:, :, -100:, -100:] = rel_in
        rel_mat = rel_mat.int()

        mask = []
        for idx, l in enumerate(length):
            mask_l = torch.zeros(x.shape[1], x.shape[1])
            mask_l[: 1+l, :1+l] = 1
            mask.append(mask_l)
        mask = torch.stack(mask, dim=0).int()
        
        # Forward pass
        x = self.decoder.tfm(x, rel_mat, mask=mask)

        # Decoder atr head
        out = self.decoder.out(x)[:, 1:, :][output_mask > 0]

        # Slice the outputs
        slices = []
        start = 0
        for i, size in enumerate(self.decoder_splits):
            slices.append(out[:, start: start + size])
            start += size

        return slices
        
    def forward(self):
        # implement me
        raise
        return