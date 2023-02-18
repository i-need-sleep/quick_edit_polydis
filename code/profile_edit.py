import os
import argparse
import datetime
import random
import copy

import pretty_midi
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from dataset import LoaderWrapper
from models.edit_musebert import EditMuseBERT
from utils.data_utils import prep_batch, prep_batch_inference, onset_pitch_duration_prettymidi_notes
import utils.rules
from torch.profiler import profile, record_function, ProfilerActivity
from models.musebert.note_attribute_repr import decode_atr_mat_to_nmat

def profile_edit(args):

    # Device
    torch.manual_seed(21)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Make loaders
    if args.batched:
        wrapper = LoaderWrapper(48, 32, edit_scheme=args.edit_scheme)
    else:
        wrapper = LoaderWrapper(1, 1, edit_scheme=args.edit_scheme)
        
    train_loader = wrapper.get_loader(split='train')
    dev_loader = wrapper.get_loader(split='dev')
    print(f'Training laoder size: {len(train_loader)}')
    print(f'Dev laoder size: {len(dev_loader)}')
    print(f'Dev #songs: {- dev_loader.dataset.split_idx}')

    # Set the rule set
    if args.identity_rule:
        wrapper.collate.rule = utils.rules.identity

    # Setup training
    model = EditMuseBERT(device, wrapper, include_original_notes=args.include_original_notes).to(device)

    # Load from checkpoints
    print(f'loading checkpoint: {args.checkpoint}')
    loaded = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(loaded['model_state_dict'])

    # Make a chord prog
    chd_types = {
        'M3': [0, 4, 7],
        'm3': [0, 3, 7],
        'A3': [0, 4, 8],
        'd3': [0, 3, 6],
        'M7': [0, 4, 7, 11],
        'm7': [0, 3, 7, 10],
        'D7': [0, 4, 7, 10],
    }

    def make_chd(root, chroma, bass):
        # root / bass: indices 0-11. Remember bass is relative
        # chroma: list of indices 0-11 (absolute)
        out = [0 for _ in range(36)]
        out[root] = 1
        out[bass + 24] = 1
        for c in chroma:
            out[c + 12] = 1 
        return out

    def make_prog(prog_text):
        out = []
        for chd_text in prog_text:
            root = chd_text[0]
            bass = chd_text[2]
            chroma = chd_types[chd_text[1]]
            out.append(make_chd(root, chroma, bass))
        return out

    cmat = make_prog([
        [9, 'm3', 0],
        [9, 'm3', 7],
        [5, 'M3', 0],
        [5, 'M3', 0],
        [5, 'M3', 0],
        [7, 'M3', 0],
        [8, 'd3', 0],
        [9, 'M3', 0],
        ])

    cmat = torch.tensor(cmat).to(device).float().unsqueeze(0)

    # Set up the profiler
    activities = activities=[ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        for _ in range(args.n_iter):
            with record_function("loop"):
                with record_function("Load data"):
                    # Get a texture from PoP909
                    _, _, pr_mat, ptree, _ = train_loader.dataset.polydis_dataset[random.randint(0, len(train_loader.dataset.polydis_dataset) - 1)]
                    pr_mat = torch.tensor(pr_mat).to(device).float()
                    ptree = torch.tensor(ptree).to(device)[0]

                with record_function("Polydis"):
                    # Polydis oracle
                    ptree_polydis = wrapper.collate.polydis.swap(pr_mat, pr_mat, cmat, cmat, fix_rhy=True, fix_chd=False)[0]
                    _, notes_polydis = wrapper.collate.polydis.decoder.grid_to_pr_and_notes(ptree_polydis.astype(int))

                
                with record_function("Edit Pipeline"):
                    with record_function("Prep data"):
                        with record_function("Recover original notes"):
                            # Original notes
                            _, notes_original = wrapper.collate.polydis.decoder.grid_to_pr_and_notes(ptree.cpu().numpy().astype(int))
                            notes_original_ = copy.deepcopy(notes_original)

                        with record_function("Apply rules"):
                            # Apply rule-based approximations to the input notes
                            notes_rule = wrapper.collate.rule(notes_original_, cmat[0])
                            notes_rule_ = copy.deepcopy(notes_rule)

                        with record_function("Convert for MuseBERT"):
                            # Convert notes for MuseBERT input
                            notes_out_line, _, _, _, _ = wrapper.collate.editor.get_edits(notes_rule, notes_polydis)
                            atr, _, cpt_rel, _, _, length = wrapper.collate.converter.convert(notes_out_line)
                            if args.altered_atr_original_rel:
                                notes_original__ = utils.data_utils.prettymidi_notes_to_onset_pitch_duration(notes_original)
                                _, _, cpt_rel, _, _, _ = wrapper.collate.converter.convert(notes_original__)


                    with record_function("Edit Models"):
                        # Run the edit models
                        atr = torch.tensor(atr).to(device).unsqueeze(0)
                        cpt_rel = torch.tensor(cpt_rel).to(device).unsqueeze(0)
                        length = [length]

                        chd = cmat
                        editor_in = [atr, cpt_rel, length]

                        with record_function("Chord encoder"):
                            z_chd = model.encode_chd(chd)

                        with record_function("Edit model"):
                            edits_out, n_inserts_out = model.encode(editor_in, z_chd, mask_by_line=True)

                        with record_function("Decode Edits"):
                            decoder_in, decoder_output_mask, context_notes = model.decode_editor_out(edits_out, n_inserts_out, editor_in)

                        with record_function("Insertion model"):
                            slicess = model.decode(decoder_in, z_chd, decoder_output_mask, per_line=True)
                        
                        with record_function("Decode insertions"):
                            out = []
                            for i, slices in enumerate(slicess):
                            
                                inserted_atr = model.slices_to_atr(slices)
                                if len(inserted_atr) == 0:
                                    inserted_notes = []
                                elif len(inserted_atr) == 1:
                                    inserted_atr *= 2
                                    inserted_notes = decode_atr_mat_to_nmat(np.array(inserted_atr)).tolist()[: 1]
                                else:
                                    inserted_notes = decode_atr_mat_to_nmat(np.array(inserted_atr)).tolist()
                            
                                out.append(context_notes[i] + inserted_notes)
    # Print/Save results
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    prof.export_chrome_trace(f"../results/traces/trace_{device}_{args.n_iter}iter.json")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_iter', default=3, type=int)
    parser.add_argument('--checkpoint', default='../results/checkpoints/mfmc/mfmc_altered_atr_original_rel.bin', type=str) 

    # Rules
    parser.add_argument('--identity_rule', action='store_true')

    # Edit operation sets
    parser.add_argument('--edit_scheme', default='mfmc', type=str) 

    # Model input
    parser.add_argument('--include_original_notes', action='store_true')
    parser.add_argument('--swap_original_rules', action='store_true')
    parser.add_argument('--altered_atr_original_rel', action='store_true')

    # Debug
    parser.add_argument('--debug', action='store_true')

    # Profile
    parser.add_argument('--batched', action='store_true')

    args = parser.parse_args()

    profile_edit(args)