import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import models.polydis.dataset as dtst
from models.polydis.model import DisentangleVAE
from models.musebert.note_attribute_repr import NoteAttributeAutoEncoder
from models.musebert.note_attribute_corrupter import SimpleCorrupter
from models.musebert.curriculum_preset import default_autoenc_dict

import utils.chord_sampler
import utils.rules
import utils.edits
import utils.data_utils

class EditDatast(Dataset):
    def __init__(self, split):

        # Load a Polydis dataset
        shift_low = -6
        shift_high = 6
        num_bar = 2
        contain_chord = True

        # Split the polydis dataset
        # For now, select the last 80 songs. Remember to create a proper index file later with random selection.
        fns = dtst.collect_data_fns()
        self.split_idx = -40
        if split == 'train':
            range = np.arange(len(fns))[: self.split_idx]
        elif split == 'dev':
            range = np.arange(len(fns))[self.split_idx: ]
        else:
            raise

        self.polydis_dataset = dtst.wrap_dataset(fns, range, shift_low, shift_high,
                                num_bar=num_bar, contain_chord=contain_chord)

        # Draw chords from the full 909 dataset
        full_909 = dtst.wrap_dataset(fns, np.arange(len(fns)), shift_low, shift_high,
                                num_bar=num_bar, contain_chord=contain_chord)
        self.sampler909 = utils.chord_sampler.Sampler909(full_909)

        self.chord_sampling_method = '909_prog'

    def __len__(self):
        return len(self.polydis_dataset)

    def __getitem__(self, index):
        _, _, pr_mat, ptree, _ = self.polydis_dataset[index]
        pr_mat = torch.tensor(pr_mat)
        ptree = torch.tensor(ptree)

        if self.chord_sampling_method == 'triad':
            # Randomly sample triads in the root position
            step_method = utils.chord_sampler.random_triad_step
            chords = utils.chord_sampler.gen_chords(step_method)
        elif self.chord_sampling_method == 'any':
            # Randomly sample any chord with any chorma at with any bass
            step_method = utils.chord_sampler.random_any_step
            chords = utils.chord_sampler.gen_chords(step_method)
        elif self.chord_sampling_method == '909_chord':
            # Sample chords by their frequency in Pop909
            step_method = self.sampler909.draw_chord
            chords = utils.chord_sampler.gen_chords(step_method)
        elif self.chord_sampling_method == '909_prog':
            # Sample a sequence of chords by their frequency in Pop909\
            chords = self.sampler909.draw_prog()
        else:
            raise

        return pr_mat, ptree, chords

class Collate(object):
    def __init__(self, editor):
        # Load a Polydis model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.polydis = DisentangleVAE.init_model(self.device)
        model_path = '../pretrained/polydis.pt'  
        self.polydis.load_model(model_path, map_location=self.device)
        self.polydis.eval()
        print(f'Polydis loaded from: {model_path}')

        # Initialise the edit operation set
        self.editor = editor
        self.max_n_inserts = 0

        # Converter to MuseBERT's input format
        self.converter = Note2MuseBERTConverter()
        self.rule = utils.rules.nearest_neighbour

    def __call__(self, batch):
        # Stack the batch
        for idx, line in enumerate(batch):
            pr_mat_line, ptree_line, chords_line = line
            chords_line = chords_line.unsqueeze(0)
            if idx == 0:
                pr_mat = pr_mat_line
                ptree = ptree_line
                chords = chords_line
            else:
                pr_mat = torch.cat((pr_mat, pr_mat_line), dim=0)
                ptree = torch.cat((ptree, ptree_line), dim=0)
                chords = torch.cat((chords, chords_line), dim=0)
                
        # Apply Polydis
        pr_mat = pr_mat.float().to(self.device)
        chords = chords.float().to(self.device)
        ptree_polydis = self.polydis.swap(pr_mat, pr_mat, chords, chords, fix_rhy=True, fix_chd=False)
        
        # Process each line
        notes_out, pitch_changes, n_inserts, inserts = [], [], [], []
        atr, cpt_atr, cpt_rel, mask, inds, length = [], [], [], [], [], []
        atr_dec, cpt_atr_dec, cpt_rel_dec, length_dec, output_mask_dec = [], [], [], [], []
        atr_original, rel_original = [], []
        notes_ref = []
        
        for line_idx in range(pr_mat.shape[0]):
            # Notes after HST
            _, notes_polydis = self.polydis.decoder.grid_to_pr_and_notes(ptree_polydis[line_idx].astype(int))

            # Original notes
            _, notes = self.polydis.decoder.grid_to_pr_and_notes(ptree[line_idx].numpy().astype(int))

            # Apply rule-based approximations to the input notes
            notes_rule = self.rule(notes, chords[line_idx])

            # Derive edits 
            notes_out_line, pitch_changes_line, n_inserts_line, inserts_line, decoder_notes_in_line = self.editor.get_edits(notes_rule, notes_polydis)

            # Truncate
            notes_out_line = notes_out_line[: self.converter.pad_length - 2]
            pitch_changes_line = pitch_changes_line[: self.converter.pad_length - 2]
            decoder_notes_in_line = decoder_notes_in_line[: self.converter.pad_length - 2]
            
            # The tokens the decoder will predict
            decoder_notes_out_line = self.editor.prep_decoder_notes(inserts_line, decoder_notes_in_line)

            # Truncate
            decoder_notes_out_line = decoder_notes_out_line[: self.converter.pad_length - len(decoder_notes_in_line)]
            
            notes_out.append(notes_out_line)
            pitch_changes.append(pitch_changes_line)
            n_inserts.append(n_inserts_line)
            inserts.append(inserts_line)

            # Convert notes for MuseBERT input
            atr_l, cpt_atr_l, cpt_rel_l, mask_l, inds_l, length_l = self.converter.convert(notes_out_line)
            
            atr.append(atr_l)
            cpt_atr.append(cpt_atr_l)
            cpt_rel.append(cpt_rel_l)
            mask.append(mask_l)
            inds.append(inds_l)
            length.append(length_l)
            
            # Also convert the original notes
            notes_original = utils.data_utils.prettymidi_notes_to_onset_pitch_duration(notes)
            notes_original = notes_original[: self.converter.pad_length - 2]
            atr_original_l, _, rel_original_l, _, _, _ = self.converter.convert(notes_original)

            atr_original.append(atr_original_l)
            rel_original.append(rel_original_l)

            # Also for the decoder
            atr_dec_l, cpt_atr_dec_l, cpt_rel_dec_l, length_dec_l, output_mask_dec_l = self.converter.convert_for_decoder(decoder_notes_in_line, decoder_notes_out_line)

            atr_dec.append(atr_dec_l)
            cpt_atr_dec.append(cpt_atr_dec_l)
            cpt_rel_dec.append(cpt_rel_dec_l)
            length_dec.append(length_dec_l)
            output_mask_dec.append(output_mask_dec_l)

            if self.max_n_inserts < max(n_inserts_line):
                self.max_n_inserts = max(n_inserts_line) # this is around 6 in pop909

            # Output reference
            notes_ref_line = utils.data_utils.prettymidi_notes_to_onset_pitch_duration(notes_polydis)
            notes_ref.append(notes_ref_line)
                
        atr = np.array(atr)
        cpt_atr = np.array(cpt_atr)
        cpt_rel = np.array(cpt_rel)
        mask = np.array(mask)
        inds = np.array(inds)

        atr_dec = np.array(atr_dec)
        cpt_atr_dec = np.array(cpt_atr_dec)
        cpt_rel_dec = np.array(cpt_rel_dec)
        output_mask_dec = torch.stack(output_mask_dec, dim=0).squeeze(1)

        atr_original = np.array(atr_original)
        rel_original = np.array(rel_original)
            
        return {
            'chords': chords,

            'notes_out': notes_out,
            'pitch_changes': pitch_changes,
            'n_inserts': n_inserts,
            'inserts': inserts,

            'atr': atr,
            'cpt_atr': cpt_atr,
            'cpt_rel': cpt_rel,
            'mask': mask,
            'inds': inds,
            'length': length,

            'atr_dec': atr_dec,
            'cpt_atr_dec': cpt_atr_dec,
            'cpt_rel_dec': cpt_rel_dec,
            'length_dec': length_dec,
            'output_mask_dec': output_mask_dec,

            'notes_ref': notes_ref,

            'atr_original': atr_original,
            'rel_original': rel_original,
        }

class LoaderWrapper(object):

    def __init__(self, batch_size, batch_size_dev, edit_scheme='mfmc', shuffle=True):
        if edit_scheme == 'default':
            self.editor = utils.edits.DefaultEditSet()
        elif edit_scheme == 'mfmc':
            self.editor = utils.edits.MFMCEditSet()
        else:
            raise
        
        self.collate = Collate(self.editor)
        self.train_set = EditDatast('train')
        self.dev_set = EditDatast('dev')

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, collate_fn=self.collate, shuffle=shuffle)
        self.dev_loader = DataLoader(self.dev_set, batch_size=batch_size_dev, collate_fn=self.collate, shuffle=shuffle)

    def get_loader(self, split):
        if split == 'train':
            return self.train_loader
        elif split == 'dev':
            return self.dev_loader
        else:
            raise

# Wrap notes for musebert input
class Note2MuseBERTConverter():
    def __init__(self):
        self.pad_length = 200 

        autoenc_dict = default_autoenc_dict
        autoenc_dict['nmat_pad_length'] = self.pad_length
        autoenc_dict['atr_mat_pad_length'] = self.pad_length

        self.repr_autoenc = NoteAttributeAutoEncoder(**default_autoenc_dict)

        # Set up a corruptor that does not corrupt
        corruptor_dict  = {
            'corrupt_col_ids': (1,),
            'pad_length': self.pad_length,
            'mask_ratio': 0.,
            'unchange_ratio': 0.,
            'unknown_ratio': 0.,
            'relmat_cpt_ratio': 0.
        }
        self.corrupter = SimpleCorrupter(**corruptor_dict)

        # atr masking
        self.unknown_values = (9, 7, 7, 3, 12, 5, 8)
        self.atr_cols = [2, 3, 4, 5, 6] # Pitches/durations (to be masked as unknown for the decoder ref)

        # rel masking
        self.rel_mask = 4

    def convert(self, notes):

        # Zero-pad the notes up self.pad_length
        notes_len = len(notes)
        if notes_len > self.pad_length:
            print('TOO LOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOONG!??')
            raise

        while len(notes) < self.pad_length:
            notes.append([0, 0, 0])
        notes = np.array(notes)

        # Make the factorised atr mat and ctp mat
        self.repr_autoenc.fast_mode()
        self.corrupter.fast_mode()
        
        atr_mat, notes_len = self.repr_autoenc.encode(notes, notes_len)
        if notes_len < 2:
            # Dirty fix for rare, empty cases
            cpt_atrmat = atr_mat
            inds = [1],
            cpt_relmat = np.zeros((4, self.pad_length, self.pad_length))
        else:
            try:
                cpt_atrmat, notes_len, inds, _, cpt_relmat = self.corrupter.\
                    compute_relmat_and_corrupt_atrmat_and_relmat(atr_mat, notes_len)
            except:
                print(notes)
                raise

        # square mask to mask out the pad tokens
        mask = self.generate_attention_mask(notes_len)

        return atr_mat.astype(np.int64), cpt_atrmat.astype(np.int64), \
            cpt_relmat.astype(np.int8), mask.astype(np.int8), \
            0, notes_len

    def generate_attention_mask(self, length):
        mask = np.zeros((self.pad_length, self.pad_length), dtype=np.int8)
        mask[0: length, 0: length] = 1
        return mask

    def convert_for_decoder(self, context, refs):
        # Convert the notes for NAR predictions. Mask only the pitch/duration for the ref notes.
        notes = context + refs
        notes_len = len(notes)
        
        # Build the uncorrupted atr/rel matrices
        atr, cpt_atr, cpt_rel, _, _, _ = self.convert(notes)
        
        # Corrupt cpt_atr. Mark the ref pitch/dur entries as unknown.
        for idx in range(cpt_atr.shape[0]):
            if idx >= len(context) and idx < notes_len:
                for feat_idx in self.atr_cols:
                    cpt_atr[idx, feat_idx] = self.unknown_values[feat_idx]\
        
        # Corrupt cpt_rel.
        # Onsets (o, o_bt) are known and remain unchanged.
        # Pitches (p) are masked as unknown *between the the context and ref notes*, ...
        # ...since the context notes are known and the relative pitches in ref notes can be assumed to avoid ambiguity
        cpt_rel[1, : len(context), len(context): notes_len] = self.rel_mask
        cpt_rel[1, len(context): notes_len, : len(context)] = self.rel_mask
        
        # Pitch hights (p_ht) is masked for any rel involving a ref note since it is unknown for the ref notes
        cpt_rel[3, : len(context), len(context): notes_len] = self.rel_mask
        cpt_rel[3, len(context): notes_len, : notes_len] = self.rel_mask

        # Build an output mask
        output_mask = torch.zeros(1, self.pad_length)
        output_mask[0, len(context): notes_len] = 1

        return atr, cpt_atr, cpt_rel, notes_len, output_mask


if __name__ == '__main__':
    import tqdm
    wrapper = LoaderWrapper(3, 3, edit_scheme='mfmc')
    loader = wrapper.get_loader('train')
    for batch in tqdm.tqdm(loader):
        break
    print('done')