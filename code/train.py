import os
import json
import argparse
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW

from dataset import LoaderWrapper
from models.edit_musebert import EditMuseBERT
from utils.data_utils import prep_batch

def train(args):

    print(args)

    # Device
    # torch.manual_seed(21)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Make loaders
    wrapper = LoaderWrapper(args.batch_size_dev)
    train_loader = wrapper.get_loader(split='train')
    dev_loader = wrapper.get_loader(split='dev')

    # Setup Tensorboard
    date_str = str(datetime.datetime.now())[:-7].replace(':','-')
    writer = SummaryWriter(log_dir=f'../results/runs/{args.name}/batch_size={args.batch_size}, Adam_lr={args.lr}/{date_str}' ,comment=f'{args.name}, batch_size={args.batch_size}, Adam_lr_enc={args.lr}, {date_str}')

    # Setup training
    model = EditMuseBERT(device, n_edit_types=wrapper.collate.editor.pitch_range)
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimiser = AdamW(model.parameters(), lr=args.lr)

    # Load from checkpoints
    if args.checkpoint != '':
        print('NOT IMPLEMENTED LOL TOO BAD')

    # train
    n_iter = 0
    n_prev_iter = 0
    running_loss = 0
    print('Training...')

    for epoch in range(args.n_epoch):
        for batch_idx, batch in enumerate(train_loader):

            model.train()
            optimiser.zero_grad()

            chd, editor_in, edits_ref, n_inserts_ref = prep_batch(batch, device)

            # Encoder forward pass
            z_chd = model.encode_chd(chd)
            z_pool, edits_out, n_inserts_out = model.encode(editor_in, z_chd)

            # Encoder loss
            edits_loss = criterion(edits_out, edits_ref)
            n_inserts_loss = criterion(n_inserts_out, n_inserts_ref)
            
            # Decoder forward pass
            # Teacher forcing


            # TODO: Student forcing, dynamic oracle


            
            exit()





            
            # Concatenate batches from the two domains
            input_ids = batch_chemu['input_ids'].to(device)
            output_graph = batch_chemu['output_graph'].to(device)

            cur_batch_size = input_ids.shape[0]

            input_ids_recipe = batch_recipe['input_ids'].to(device)
            output_graph_recipe = batch_recipe['output_graph'].to(device)

            input_ids = torch.cat((input_ids, input_ids_recipe), 0)
            output_graph = torch.cat((output_graph, output_graph_recipe), 0)
            
            input_graph = torch.zeros_like(output_graph).int()
            
            for refine_idx in range(args.n_refine):
                mask = input_ids != train_loader_chemu.dataset.pad_id
                        
                mask = mask.unsqueeze(2).expand(-1, -1, mask.shape[1])
                mask = mask * mask.transpose(2,1)

                x, model_mask = model.encode(input_ids, input_graph.to(device))
                s_arc, s_rel = model.pred_graph(x, model_mask)
                
                # Input graph for the next refinement iter
                if refine_idx < args.n_refine - 1:
                    input_graph = model.build_input_graph_argmax(s_arc, s_rel, mask)
                
                if not args.adv_per_token:
                    # Pool embs [batch, seq_len, emb_len] into domain embs [batch, emb_len]
                    pooling = torch.nn.AdaptiveMaxPool1d(1)
                    domain_emb = pooling(x.permute(0, 2, 1))[:, :, 0]
                    adv_pred = adversary(domain_emb).reshape(-1)

                    # Adverserial loss
                    adv_true = torch.zeros_like(adv_pred)
                    adv_true[: cur_batch_size] = 1

                else:
                    adv_pred = adversary(x)[:, :, 0]
                    adv_true = torch.zeros_like(adv_pred)
                    adv_true[: cur_batch_size, :] = 1
                    adv_pred = adv_pred[input_ids != train_loader_chemu.dataset.pad_id]
                    adv_true = adv_true[input_ids != train_loader_chemu.dataset.pad_id]
                    
                adv_loss = criterion_dom(adv_pred, adv_true)

                # Keep only arc/rel preds from the source domain
                mask = mask[: cur_batch_size, :, :]
                s_arc = s_arc[: cur_batch_size, :, :]
                s_rel = s_rel[: cur_batch_size, :, :, :]
                output_graph = output_graph[: cur_batch_size, :, :]

                s_arc_l = s_arc[mask]
                s_rel_l = s_rel[mask, :]

                true_arc = output_graph[mask]
                true_arc[true_arc > 0] = 1
                true_rel = output_graph[mask].long()

                arc_loss = criterion_arc(s_arc_l, true_arc)
                rel_loss = criterion_rel(s_rel_l, true_rel)
                loss = arc_loss + rel_loss + args.adv_scalar * adv_loss
                loss.backward()

                optimizer_bert.step()
                optimizer_nonbert.step()
                
                n_iter += 1
                writer.add_scalar('train_batch/arc_loss', arc_loss, n_iter)
                writer.add_scalar('train_batch/rel_loss', rel_loss, n_iter)
                writer.add_scalar('train_batch/adv_loss', adv_loss, n_iter)
                running_loss += loss.detach()
 
        print(f'Epoch: {epoch}')
        epoch_loss = running_loss / (n_iter - n_prev_iter)
        print(f'Training loss: {epoch_loss}')
        writer.add_scalar('Loss/train_epoch', epoch_loss, n_iter)
        n_prev_iter = n_iter
        running_loss = 0

        if epoch % 10 == 0:
            prec, recall, f1_chemu = eval(model, dev_loader_chemu, device, args.n_refine, dense_span=args.dense_span)
            writer.add_scalar('dev_chemu/prec', prec, n_iter)
            writer.add_scalar('dev_chemu/recall', recall, n_iter)
            writer.add_scalar('dev_chemu/f1', f1_chemu, n_iter)
            prec, recall, f1_recipe = eval(model, dev_loader_recipe, device, args.n_refine, dense_span=args.dense_span)
            writer.add_scalar('dev_recipe/prec', prec, n_iter)
            writer.add_scalar('dev_recipe/recall', recall, n_iter)
            writer.add_scalar('dev_recipe/f1', f1_recipe, n_iter)

            try:
                os.makedirs(f'../result/checkpoint/{args.name}')
            except:
                pass
            save_path = f'../result/checkpoint/{args.name}/batchsize{batch_size}_lr{lr}_{epoch}_{batch_idx}_{f1_chemu}_{f1_recipe}.bin'
            print(f'Saving the checkpoint at {save_path}')
            torch.save({
                'epoch': epoch,
                'step': n_iter,
                'model_state_dict': model.state_dict(),
                'adversary_state_dict': adversary.state_dict(),
                'optimizer_bert_state_dict': optimizer_bert.state_dict(),
                'optimizer_nonbert_state_dict': optimizer_nonbert.state_dict(),
                }, save_path)
                
    print('DONE !!!')

def eval(model, loader, device, n_refine, write_answer='', dense_span=False, gold_mention=False):
    model.eval()
    with torch.no_grad():
        n_rels_pred, n_rels_true, n_rels_hit = 0, 0, 0
        write_answer_out = []

        for idx, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(device)
            output_graph = batch['output_graph'].to(device)
            docs = batch['docs']
            windows = batch['windows']
            input_graph = torch.zeros_like(output_graph).int()
            
            for refine_idx in range(n_refine):
                mask = input_ids != loader.dataset.pad_id
                        
                mask = mask.unsqueeze(2).expand(-1, -1, mask.shape[1])
                mask = mask * mask.transpose(2,1)

                if gold_mention:
                    input_graph[output_graph == 1] = 1

                model_out = model(input_ids, input_graph.to(device))
                s_arc, s_rel = model_out[0], model_out[1]

                input_graph = model.build_input_graph_argmax(s_arc, s_rel, mask)

            true_mens, true_rels = graph_to_rels(output_graph, dense_span=dense_span)
            pred_mens, pred_rels = graph_to_rels(input_graph, dense_span=dense_span)

            # Eval for f1
            for i in range(len(true_mens)):
                true_men = true_mens[i]
                true_rel = true_rels[i]
                pred_men = pred_mens[i]
                pred_rel = pred_rels[i]
                
                n_rels_pred += len(pred_rel)
                n_rels_true += len(true_rel)

                for rel in pred_rel:
                    if rel in true_rel:
                        if rel[0] in pred_men.keys() and rel[0] in true_men.keys() and pred_men[rel[0]] == true_men[rel[0]]:
                            if rel[1] in pred_men.keys() and rel[1] in true_men.keys() and pred_men[rel[1]] == true_men[rel[1]]:
                                n_rels_hit += 1
                
                # Store predictions
                if write_answer != '':
                    if dense_span:
                        pred_men_ = {}
                        for key, val in pred_men.items():
                            pred_men_[key] = list(val)
                        pred_men = pred_men_

                        true_men_ = {}
                        for key, val in true_men.items():
                            true_men_[key] = list(val)
                        true_men = true_men_

                    filtered_ids = list(filter(lambda x: x!=loader.dataset.pad_id, input_ids[i]))
                    write_answer_out.append({
                        'text': loader.dataset.tokenizer.decode(filtered_ids),
                        'pred_men': pred_men,
                        'pred_rel': pred_rel,
                        'true_men': true_men,
                        'true_rel': true_rel,
                        'doc': docs[i],
                        'window': windows[i]
                    })
            
        if n_rels_pred == 0:
            n_rels_pred = 1
        if n_rels_true == 0:
            n_rels_true = 1
        prec = n_rels_hit / n_rels_pred
        recall = n_rels_hit / n_rels_true
        if prec > 0 and recall > 0:
            f1 = 2/(1/prec + 1/recall)
        else:
            f1 = 0
        print(f'#pred: {n_rels_pred}, #true:{n_rels_true}, #hits:{n_rels_hit}')
        print(f'F1: {f1}, prec: {prec}, recall: {recall}')
        
        if write_answer != '':
            with open(f'../result/output/{write_answer}.json', 'w') as f:
                json.dump(write_answer_out, f)

        return prec, recall, f1
            
def rels_to_dict(ids, mens, rels, loader, dense_span=False):
    out = []

    def to_text(tokenizer, ids, indices):
        last_indice = None
        line_out = []
        out = []

        for index in indices:
            if last_indice != None and last_indice != index-1:
                out.append(line_out)
                line_out = []
            line_out.append(ids[index])
            last_indice = index
        if line_out != []:
            out.append(line_out)
            
        str_out = ''
        for line in out:
            str_out += tokenizer.decode(line) + ' '
        str_out = str_out.strip()

        return str_out

    for rel in rels:
        if rel[0] in mens.keys() and rel[1] in mens.keys():
            if dense_span:
                out.append({
                    'subj': to_text(loader.dataset.tokenizer, ids, mens[rel[0]]),
                    'obj': to_text(loader.dataset.tokenizer, ids, mens[rel[1]]),
                    'type': loader.dataset.rel_types[rel[2]]
                })
            else:
                out.append({
                    'subj': loader.dataset.tokenizer.decode(ids[rel[0]: mens[rel[0]]+1]),
                    'obj': loader.dataset.tokenizer.decode(ids[rel[1]: mens[rel[1]]+1]),
                    'type': loader.dataset.rel_types[rel[2]]
                })
    return out

def graph_to_rels(graphs, dense_span=False):
    men_out = []
    rel_out = []
    for idx in range(graphs.shape[0]):
        graph = graphs[idx, :, :]

        # mentions {start: end, ...}
        # Dense mentions {start: [included indices]}
        # rels [[subj, obj, type], ...]\
        mens = {}
        rels = []

        men_list = (graph == 1).nonzero().tolist()
        for men in men_list:
            if dense_span:
                if men[1] not in mens.keys():
                    mens[men[1]] = set()
                if men[0] >= men[1]:
                    mens[men[1]].add(men[0])

            else:
                mens[men[1]] = men[0]

        for rel_type in range(2, 10):
            rel_list = (graph == rel_type).nonzero().tolist()
            for rel in rel_list:
                rels.append([rel[0], rel[1], rel_type])

        men_out.append(mens)
        rel_out.append(rels)
    return men_out, rel_out

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