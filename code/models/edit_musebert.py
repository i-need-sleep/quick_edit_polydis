import torch
import numpy as np

from models.polydis.ptvae import RnnEncoder
from models.polydis.amc_dl.torch_plus.train_utils import get_zs_from_dists
from models.musebert.musebert_model import MuseBERT
from models.musebert.note_attribute_repr import decode_atr_mat_to_nmat

class EditMuseBERT(torch.nn.Module):
    def __init__(self, device, wrapper, pretrained_path='../pretrained/musebert.pt', n_edit_types=128, max_n_inserts=16, n_decoder_layers=2):
        super(EditMuseBERT, self).__init__()
        
        self.wrapper = wrapper
        self.device = device
        self.pad_len = self.wrapper.collate.converter.pad_length
        self.max_n_inserts = max_n_inserts
        
        # Load a pretrained MuseBERT encoder
        self.encoder = MuseBERT.init_model(loss_inds=(0, 1, 2, 3, 4, 5, 6), relation_vocab_sizes=(5, 5, 5, 5)).to(device)
        self.encoder.load_model(pretrained_path, device)
        print(f'MuseBERT encoder loaded from: {pretrained_path}')

        self.step_embs = torch.nn.Embedding(32, 128)
        self.edit_head = torch.nn.Linear(128, n_edit_types)
        self.n_inserts_head = torch.nn.Linear(128, max_n_inserts) # num of inserts at each onset step

        # Chord encoder from Polydis (from scratch)
        self.chord_enc = RnnEncoder(36, 1024, 128).to(device)

        # Decoder
        self.decoder = MuseBERT.init_model(loss_inds=(0, 1, 2, 3, 4, 5, 6), relation_vocab_sizes=(5, 5, 5, 5), N=n_decoder_layers).to(device)
        self.decoder_splits = [9, 7, 7, 3, 12, 5, 8] # Sizes for each dim of the factorised features
    
    def encode_chd(self, chd):
        x = self.chord_enc(chd)
        z_chd = get_zs_from_dists([x], False)[0]
        return z_chd

    def encode(self, editor_in, z_chd, mask_by_line=False):
        [data_in, rel_mat_in, length] = editor_in

        # Embed the time step tokens (for predicting n_inserts)
        onset_steps = torch.tensor([[i for i in range(32)] for j in range(data_in.shape[0])])
        onset_embs = self.step_embs(onset_steps.to(self.device))

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
        rel_mat[:, :, -self.pad_len:, -self.pad_len:] = rel_mat_in
        rel_mat = rel_mat.int()

        mask = []
        for idx, l in enumerate(length):
            mask_l = torch.zeros(x.shape[1], x.shape[1])
            mask_l[: 33+l, :33+l] = 1
            mask.append(mask_l)
            edit_mask[idx, 33: 33+l] = 1
        mask = torch.stack(mask, dim=0).int()

        # Forward pass through the transformer
        x = self.encoder.tfm(x, rel_mat.to(self.device), mask=mask.to(self.device))

        if mask_by_line:
            n_inserts_out = self.n_inserts_head(x)[:, 1:33, :]
            edits_out = self.edit_head(x)
            
            edits_out_lst = []
            for i in range(edits_out.shape[0]):
                edits_out_lst.append(edits_out[i][edit_mask[i] > 0])
                
            return edits_out_lst, n_inserts_out

        print(x.shape)
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
        rel_mat[:, :, -self.pad_len:, -self.pad_len:] = rel_in
        rel_mat = rel_mat.int()

        mask = []
        for idx, l in enumerate(length):
            mask_l = torch.zeros(x.shape[1], x.shape[1])
            mask_l[: 1+l, :1+l] = 1
            mask.append(mask_l)
        mask = torch.stack(mask, dim=0).int()
        
        # Forward pass
        x = self.decoder.tfm(x, rel_mat.to(self.device), mask=mask.to(self.device))

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

    def inference(self, chd ,editor_in):
        z_chd = self.encode_chd(chd)
        edits_out, n_inserts_out = self.encode(editor_in, z_chd, mask_by_line=True)

        decoder_in, decoder_output_mask, context_notes = self.decode_editor_out(edits_out, n_inserts_out, editor_in)

        slices = self.decode(decoder_in, z_chd, decoder_output_mask)
        
        inserted_atr = self.slices_to_atr(slices)
        inserted_notes = decode_atr_mat_to_nmat(np.array(inserted_atr)).tolist()
        
        out = context_notes + inserted_notes
        return out
    
    def slices_to_atr(self, slices):
        out = [[0 for _ in range(7)] for _ in range(slices[0].shape[0])]
        for feat_idx, slice in enumerate(slices):
            inds = torch.max(slice, dim=1).indices.tolist()
            for step, idx in enumerate(inds):
                out[step][feat_idx] = idx
        return out


    def decode_editor_out(self, edits_out, n_inserts_out, editor_in):

        [atr, _, length] = editor_in
        cpt_atr_dec, cpt_rel_dec, length_dec, output_mask_dec = [], [], [], []

        # Let's do this line-by-line
        for line_idx in range(len(edits_out)):

            # Build the transformed notes
            notes_context = []
            nmat_line = decode_atr_mat_to_nmat(atr[line_idx][: length[line_idx]]).tolist()
            edits_line = torch.max(edits_out[line_idx], dim=1).indices

            for idx, note in enumerate(nmat_line):
                tar_pitch =  edits_line[idx] 
                if tar_pitch > 0:
                    notes_context.append([note[0], tar_pitch.item(), note[2]])

            # Build fake new notes for prediction
            notes_ref = []
            n_inserts_line = torch.max(n_inserts_out[line_idx], dim=1).indices
            for step in range(n_inserts_line.shape[0]):
                for i in range(n_inserts_line[step].item()):
                    # Manage the sequence length
                    if len(notes_ref) + len(notes_context) < self.wrapper.collate.converter.pad_length:
                        notes_ref.append([step, i+1, 1])

            # Process into decoder input
            _, cpt_atr_dec_l, cpt_rel_dec_l, length_dec_l, output_mask_dec_l = self.wrapper.collate.converter.convert_for_decoder(notes_context, notes_ref)

            cpt_atr_dec.append(cpt_atr_dec_l)
            cpt_rel_dec.append(cpt_rel_dec_l)
            length_dec.append(length_dec_l)
            output_mask_dec.append(output_mask_dec_l)
        
        cpt_atr_dec = np.array(cpt_atr_dec)
        cpt_rel_dec = np.array(cpt_rel_dec)
        output_mask_dec = torch.stack(output_mask_dec, dim=0).squeeze(1)
        
        return [torch.tensor(cpt_atr_dec).to(self.device), torch.tensor(cpt_rel_dec).to(self.device), length_dec], output_mask_dec, notes_context