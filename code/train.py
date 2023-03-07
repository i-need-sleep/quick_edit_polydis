import os
import argparse
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import LoaderWrapper
from models.edit_musebert import EditMuseBERT
from utils.data_utils import prep_batch, prep_batch_inference
import utils.rules

def train(args):

    if args.debug:
        # args.checkpoint = '../results/checkpoints/debug/batchsize32_lr1e-05_0_4999_100.bin'
        args.batch_size = 2
        args.batch_size_dev = 2
    print(args)

    # Device
    torch.manual_seed(21)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Make loaders
    wrapper = LoaderWrapper(args.batch_size, args.batch_size_dev, edit_scheme=args.edit_scheme)
    train_loader = wrapper.get_loader(split='train')
    dev_loader = wrapper.get_loader(split='dev')
    print(f'Training laoder #batches: {len(train_loader)}')
    print(f'Dev laoder #batches: {len(dev_loader)}')
    print(f'Dev #songs: {- dev_loader.dataset.split_idx}')

    # Set the rule set
    if args.identity_rule:
        wrapper.collate.rule = utils.rules.identity

    # Setup Tensorboard
    date_str = str(datetime.datetime.now())[:-7].replace(':','-')
    writer = SummaryWriter(log_dir=f'../results/runs/{args.name}/batch_size={args.batch_size}, Adam_lr={args.lr}/{date_str}' ,comment=f'{args.name}, batch_size={args.batch_size}, Adam_lr_enc={args.lr}, {date_str}')

    # Setup training
    model = EditMuseBERT(device, wrapper, include_original_notes=args.include_original_notes, from_scratch=args.from_scratch).to(device)
    
    # CE losses. Use a weigheed loss for edits
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Load from checkpoints
    if args.checkpoint != '':
        print(f'loading checkpoint: {args.checkpoint}')
        loaded = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(loaded['model_state_dict'])
        if not args.eval:
            optimiser.load_state_dict(loaded['optimiser_state_dict'])
            optimiser.param_groups[0]['capturable'] = True

    # train
    n_iter = 0
    n_prev_iter = 0
    running_loss = 0
    print('Training...')

    for epoch in range(args.n_epoch):
        # Curriculum. Change me
        wrapper.train_loader.dataset.chord_sampling_method = '909_prog'

        for batch_idx, batch in enumerate(train_loader):
            
            if not args.eval:

                model.train()
                optimiser.zero_grad()

                chd, editor_in, edits_ref, n_inserts_ref, decoder_in, decoder_ref, decoder_out_mask = prep_batch(batch, device, include_original_notes=args.include_original_notes, swap_original_rules=args.swap_original_rules, altered_atr_original_rel=args.altered_atr_original_rel)

                # Encoder forward pass
                z_chd = model.encode_chd(chd)
                _, edits_out, n_inserts_out = model.encode(editor_in, z_chd)

                # Encoder loss
                edits_loss = criterion(edits_out, edits_ref)
                n_inserts_loss = criterion(n_inserts_out, n_inserts_ref)

                # Decoder forward pass
                # Teacher forcing
                # TODO: Student forcing, dynamic oracle?
                decoder_out = model.decode(decoder_in, z_chd, decoder_out_mask) # [[ref_len, feat_dim],...]

                # Decoder loss
                decoder_loss = 0
                for i in range(decoder_ref.shape[1]):
                    decoder_loss += criterion(decoder_out[i], decoder_ref[:, i]) / decoder_ref.shape[1]

                # Backward pass
                total_loss = edits_loss + n_inserts_loss + decoder_loss
                
                total_loss.backward()
                optimiser.step()
                    
                writer.add_scalar('loss/edits_loss', edits_loss, n_iter)
                writer.add_scalar('loss/n_inserts_loss', n_inserts_loss, n_iter)
                writer.add_scalar('loss/decoder_loss', decoder_loss, n_iter)
                writer.add_scalar('loss/total_loss', total_loss, n_iter)
                running_loss += total_loss.detach()
            
            n_iter += 1

            # Print running losses
            if n_iter % 100 == 0:
                print(n_iter)
                step_loss = running_loss / (n_iter - n_prev_iter)
                print(f'Training loss: {step_loss}')
                n_prev_iter = n_iter
                running_loss = 0

            # Save the checkpoint
            if n_iter % 1000 == 0:
                try:
                    os.makedirs(f'../results/checkpoints/{args.name}')
                except:
                    pass
                save_path = f'../results/checkpoints/{args.name}/batchsize{args.batch_size}_lr{args.lr}_{epoch}_{batch_idx}.bin'
                print(f'Saving the checkpoint at {save_path}')
                torch.save({
                    'epoch': epoch,
                    'step': n_iter,
                    'model_state_dict': model.state_dict(),
                    'optimiser_state_dict': optimiser.state_dict(),
                    }, save_path)

            # Eval
            if n_iter % 5000 == 0 or args.eval:
                eval(model, dev_loader, device, args)
                    
    print('DONE !!!')

def eval(model, loader, device, args):
    loader.dataset.chord_sampling_method = '909_prog'
    
    model.eval()
    with torch.no_grad():
        n_preds, n_refs, n_hits = 0, 0, 0

        for idx, batch in enumerate(loader):
            
            # notes_ref: [[note sequence: [start, pitch, dur], ...], ...]
            chd, editor_in, notes_ref, notes_rule = prep_batch_inference(batch, device, include_original_notes=args.include_original_notes, swap_original_rules=args.swap_original_rules, altered_atr_original_rel=args.altered_atr_original_rel)
            if not args.eval_rules:
                notes_pred = model.inference(chd, editor_in)
            else:
                notes_pred = notes_rule
                
            for i in range(len(notes_ref)):
                n_pred, n_ref, n_hit = eval_notes_hits_timestep(notes_pred[i], notes_ref[i])
                n_preds += n_pred
                n_refs += n_ref
                n_hits += n_hit

        # Eval for f1
        if n_pred == 0:
            n_pred = 1
        if n_ref == 0:
            n_ref = 1
        prec = n_hit / n_pred
        recall = n_hit / n_ref
        if prec > 0 and recall > 0:
            f1 = 2/(1/prec + 1/recall)
        else:
            f1 = 0
        print(f'#pred: {n_pred}, #true:{n_ref}, #hits:{n_hit}')
        print(f'F1: {f1}, prec: {prec}, recall: {recall}')
        
        return prec, recall, f1

def eval_notes_hits(notes_pred, notes_ref):
    n_hits = 0
    for note_pred in notes_pred:
        [start, pitch, dur] = note_pred
        for i in range(len(notes_ref)):
            note_ref = notes_ref[i]
            if note_ref[0] == start and note_ref[1] == pitch and note_ref[2] == dur:
                n_hits += 1
                notes_ref.pop(i)
                break
    return n_hits

def notes_break_down_to_steps(notes):
    out = []
    for note in notes:
        for step in range(note[0], note[2]):
            out.append([note[1], step])
    return out

def eval_notes_hits_timestep(notes_pred, notes_ref):
    # notes: [[start, pitch, duration], ...]

    # break down into [pitch, step] for all steps the pitch is played
    pieces_pred = notes_break_down_to_steps(notes_pred)
    pieces_ref = notes_break_down_to_steps(notes_ref)

    n_pred = len(pieces_pred)
    n_ref = len(pieces_ref)
    n_hit = 0
    
    for piece in pieces_pred:
        if piece in pieces_ref:
            n_hit += 1
    return n_pred, n_ref, n_hit

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='unnamed')

    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--batch_size_dev', default=32, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--n_epoch', default=1000, type=int)
    parser.add_argument('--checkpoint', default='', type=str) 

    # Eval
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_rules', action='store_true')

    # Rules
    parser.add_argument('--identity_rule', action='store_true')

    # Edit operation sets
    parser.add_argument('--edit_scheme', default='mfmc', type=str) 

    # Model input
    parser.add_argument('--include_original_notes', action='store_true')
    parser.add_argument('--swap_original_rules', action='store_true')
    parser.add_argument('--from_scratch', action='store_true')
    parser.add_argument('--altered_atr_original_rel', action='store_true')

    # Debug
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    train(args)