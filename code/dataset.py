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
        split_idx = -80
        if split == 'train':
            range = np.arange(len(fns))[: split_idx]
        elif split == 'dev':
            range = np.arange(len(fns))[split_idx: ]
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
        if self.chord_sampling_method == 'any':
            # Randomly sample any chord with any chorma at with any bass
            step_method = utils.chord_sampler.random_any_step
            chords = utils.chord_sampler.gen_chords(step_method)
        elif self.chord_sampling_method == '909_chord':
            # Sample chords by their frequency in Pop909
            step_method = self.sampler909.draw_chord
            chords = utils.chord_sampler.gen_chords(step_method)
        elif self.chord_sampling_method == '909_prog':
            # Sample a sequence of chords by their frequency in Pop909
            chords = self.sampler909.draw_prog()

        return pr_mat, ptree, chords

class Collate(object):
    def __init__(self, editor):
        # Load a Polydis model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.polydis = DisentangleVAE.init_model(device)
        model_path = '../pretrained/polydis.pt'  
        self.polydis.load_model(model_path, map_location=device)
        self.polydis.eval()
        print(f'Polydis loaded from: {model_path}')

        # Initialise the edit operation set
        self.editor = editor
        self.max_n_inserts = 0

        # Converter to MuseBERT's input format
        self.converter = Note2MuseBERTConverter()

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
        pr_mat = pr_mat.float()
        ptree_polydis = self.polydis.swap(pr_mat, pr_mat, chords, chords, fix_rhy=True, fix_chd=False)
        
        # Process each line
        notes_out, pitch_changes, n_inserts, inserts = [], [], [], []
        atr, cpt_atr, cpt_rel, mask, inds, length = [], [], [], [], [], []
        for line_idx in range(pr_mat.shape[0]):
            # Notes after HST
            _, notes_polydis = self.polydis.decoder.grid_to_pr_and_notes(ptree_polydis[line_idx].astype(int))

            # Original notes
            _, notes = self.polydis.decoder.grid_to_pr_and_notes(ptree[line_idx].numpy().astype(int))

            # Apply rule-based approximations to the input notes
            notes_rule = utils.rules.nearest_neighbour(notes, chords[line_idx])

            # Derive edits 
            notes_out_line, pitch_changes_line, n_inserts_line, inserts_line, decoder_notes_in_line = self.editor.get_edits(notes_rule, notes_polydis)
            
            # Given inserts, n_inserts and decoder inputs, build input/output tokens
            decoder_notes_out_line = self.editor.prep_decoder_notes(inserts_line, decoder_notes_in_line) # implement me
            
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

            if self.max_n_inserts < max(n_inserts_line):
                self.max_n_inserts = max(n_inserts_line) # this is around 6 in pop909
                
        atr = np.array(atr)
        cpt_atr = np.array(cpt_atr)
        cpt_rel = np.array(cpt_rel)
        mask = np.array(mask)
        inds = np.array(inds)
            
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
            'length': length
        }

class LoaderWrapper(object):
    def __init__(self, batch_size, shuffle=True):
        self.editor = utils.edits.DefaultEditSet()
        self.collate = Collate(self.editor)
        self.train_set = EditDatast('train')
        self.dev_set = EditDatast('dev')

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, collate_fn=self.collate, shuffle=shuffle)
        self.dev_loader = DataLoader(self.dev_set, batch_size=batch_size, collate_fn=self.collate, shuffle=shuffle)

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
        self.pad_length = 100 
        self.repr_autoenc = NoteAttributeAutoEncoder(**default_autoenc_dict)

        # Set up a corruptor that does not corrupt
        corruptor_dict  = {
            'corrupt_col_ids': (1,),
            'pad_length': 100,
            'mask_ratio': 0.,
            'unchange_ratio': 0.,
            'unknown_ratio': 0.,
            'relmat_cpt_ratio': 0.
        }
        self.corrupter = SimpleCorrupter(**corruptor_dict)

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
        cpt_atrmat, notes_len, inds, _, cpt_relmat = self.corrupter.\
            compute_relmat_and_corrupt_atrmat_and_relmat(atr_mat, notes_len)

        # square mask to mask out the pad tokens
        mask = self.generate_attention_mask(notes_len)

        return atr_mat.astype(np.int64), cpt_atrmat.astype(np.int64), \
            cpt_relmat.astype(np.int8), mask.astype(np.int8), \
            inds.astype(bool), notes_len

    def generate_attention_mask(self, length):
        mask = np.zeros((self.pad_length, self.pad_length), dtype=np.int8)
        mask[0: length, 0: length] = 1
        return mask

if __name__ == '__main__':
    wrapper = LoaderWrapper(3)
    loader = wrapper.get_loader('train')
    for batch in loader:
        break