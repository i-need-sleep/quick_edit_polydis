import os
import json
import argparse
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import LoaderWrapper
from models.edit_musebert import EditMuseBERT
from utils.data_utils import prep_batch, prep_batch_inference

def train(args):

    print(args)

    # Device
    # torch.manual_seed(21)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Make loaders
    wrapper = LoaderWrapper(args.batch_size, args.batch_size_dev)
    train_loader = wrapper.get_loader(split='train')
    dev_loader = wrapper.get_loader(split='dev')

    # Setup Tensorboard
    date_str = str(datetime.datetime.now())[:-7].replace(':','-')
    writer = SummaryWriter(log_dir=f'../results/runs/{args.name}/batch_size={args.batch_size}, Adam_lr={args.lr}/{date_str}' ,comment=f'{args.name}, batch_size={args.batch_size}, Adam_lr_enc={args.lr}, {date_str}')

    # Setup training
    model = EditMuseBERT(device, wrapper, n_edit_types=wrapper.collate.editor.pitch_range).to(device)
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
 
        print(f'Epoch: {epoch}')
        epoch_loss = running_loss / (n_iter - n_prev_iter)
        print(f'Training loss: {epoch_loss}')
        writer.add_scalar('loss/epoch_loss', epoch_loss, n_iter)
        n_prev_iter = n_iter
        running_loss = 0

        if epoch % 5 == 0:
            prec, recall, f1 = eval(model, dev_loader, device)
            writer.add_scalar('dev/prec', prec, n_iter)
            writer.add_scalar('dev/recall', recall, n_iter)
            writer.add_scalar('dev/f1', f1, n_iter)

            if f1 > best_f1:
                best_f1 = f1
                try:
                    os.makedirs(f'../result/checkpoint/{args.name}')
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
    loader.dataset.chord_sampling_method = '909_chord'
    
    model.eval()
    with torch.no_grad():
        n_pred, n_ref, n_hit = 0, 0, 0

        for idx, batch in enumerate(loader):
            
            chd, editor_in, notes_ref = prep_batch_inference(batch, device)
            notes_pred = model.inference(chd, editor_in)

            n_pred += len(notes_pred)
            n_ref += len(notes_ref)
            n_hit = eval_notes_hits(notes_pred, notes_ref)

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

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--batch_size_dev', default=16, type=int)
    parser.add_argument('--lr', default=1e-6, type=float)
    parser.add_argument('--n_epoch', default=1000, type=int)
    parser.add_argument('--checkpoint', default='', type=str) 

    args = parser.parse_args()

    train(args)