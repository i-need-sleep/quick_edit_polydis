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
        args.checkpoint = '../results/checkpoints/debug/batchsize32_lr1e-05_0_4999_100.bin'
        args.batch_size = 2
        args.batch_size_dev = 2
    print(args)

    # Device
    torch.manual_seed(21)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Make loaders
    wrapper = LoaderWrapper(args.batch_size, args.batch_size_dev)
    train_loader = wrapper.get_loader(split='train')
    dev_loader = wrapper.get_loader(split='dev')
    print(f'Training laoder size: {len(train_loader)}')
    print(f'Dev laoder size: {len(dev_loader)}')
    print(f'Dev #songs: {- dev_loader.dataset.split_idx}')

    # Set the rule set
    if args.identity_rule:
        wrapper.collate.rule = utils.rules.identity

    # Setup Tensorboard
    date_str = str(datetime.datetime.now())[:-7].replace(':','-')
    writer = SummaryWriter(log_dir=f'../results/runs/{args.name}/batch_size={args.batch_size}, Adam_lr={args.lr}/{date_str}' ,comment=f'{args.name}, batch_size={args.batch_size}, Adam_lr_enc={args.lr}, {date_str}')

    # Setup training
    if args.debug:
        model = EditMuseBERT(device, wrapper,n_edit_types=wrapper.collate.editor.pitch_range, n_decoder_layers=2).to(device)
    else:
        model = EditMuseBERT(device, wrapper,n_edit_types=wrapper.collate.editor.pitch_range).to(device)
    
    # CE losses. Use a weigheed loss for edits
    edit_weights = torch.tensor([0.1] + [1 for _ in range(127)])
    criterion_edits = torch.nn.CrossEntropyLoss(weight=edit_weights)
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Load from checkpoints
    if args.checkpoint != '':
        print(f'loading checkpoint: {args.checkpoint}')
        loaded = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(loaded['model_state_dict'])
        optimiser.load_state_dict(loaded['optimiser_state_dict'])

    # train
    n_iter = 0
    n_prev_iter = 0
    running_loss = 0
    best_f1 = 0
    print('Training...')

    for epoch in range(args.n_epoch):
        # Curriculum. Change me
        if epoch < 20:
            wrapper.train_loader.dataset.chord_sampling_method = '909_prog'
        elif epoch < 30:
            wrapper.train_loader.dataset.chord_sampling_method = '909_chord'
        else:
            wrapper.train_loader.dataset.chord_sampling_method = 'any'

        for batch_idx, batch in enumerate(train_loader):

            model.train()
            optimiser.zero_grad()

            chd, editor_in, edits_ref, n_inserts_ref, decoder_in, decoder_ref, decoder_out_mask = prep_batch(batch, device)

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
                
            n_iter += 1
            writer.add_scalar('loss/edits_loss', edits_loss, n_iter)
            writer.add_scalar('loss/n_inserts_loss', n_inserts_loss, n_iter)
            writer.add_scalar('loss/decoder_loss', decoder_loss, n_iter)
            writer.add_scalar('loss/total_loss', total_loss, n_iter)
            running_loss += total_loss.detach()

            if n_iter % 100 == 0:
                print(n_iter)
 
                step_loss = running_loss / (n_iter - n_prev_iter)
                print(f'Training loss: {step_loss}')
                n_prev_iter = n_iter
                running_loss = 0

            if n_iter % 6000 == 0 or args.debug:
                if args.debug:
                    prec, recall, f1 = eval(model, dev_loader, device)
                try:
                    prec, recall, f1 = eval(model, dev_loader, device)
                    writer.add_scalar('dev/prec', prec, n_iter)
                    writer.add_scalar('dev/recall', recall, n_iter)
                    writer.add_scalar('dev/f1', f1, n_iter)
                except:
                    print('eval died!!')
                    f1 = best_f1 + 1e-10

                if f1 > best_f1:
                    best_f1 = f1
                    try:
                        os.makedirs(f'../results/checkpoints/{args.name}')
                    except:
                        pass
                    save_path = f'../result/checkpoint/{args.name}/batchsize{args.batch_size}_lr{args.lr}_{epoch}_{batch_idx}_{f1}.bin'
                    print(f'Best f1: {best_f1}')
                    print(f'Saving the checkpoint at {save_path}')
                    torch.save({
                        'epoch': epoch,
                        'step': n_iter,
                        'model_state_dict': model.state_dict(),
                        'optimiser_state_dict': optimiser.state_dict(),
                        }, save_path)
                    
    print('DONE !!!')

def eval(model, loader, device):
    loader.dataset.chord_sampling_method = '909_prog'
    
    model.eval()
    with torch.no_grad():
        n_pred, n_ref, n_hit = 0, 0, 0

        for idx, batch in enumerate(loader):
            
            # notes_ref: [[note sequence: [start, pitch, dur], ...], ...]
            chd, editor_in, notes_ref = prep_batch_inference(batch, device)
            notes_pred = model.inference(chd, editor_in)

            for i in range(len(notes_ref)):
                n_pred += len(notes_pred[i])
                n_ref += len(notes_ref[i])
                n_hit = eval_notes_hits(notes_pred[i], notes_ref[i])    
                
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='unnamed')

    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--batch_size_dev', default=32, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--n_epoch', default=1000, type=int)
    parser.add_argument('--checkpoint', default='', type=str) 

    # Rules
    parser.add_argument('--identity_rule', action='store_true')

    # Debug
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    train(args)