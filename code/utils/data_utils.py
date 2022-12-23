import torch

def prep_batch(batch, device):

    chd = batch['chords'].to(device)

    # MuseBERT editor intput
    atr = torch.tensor(batch['atr']).to(device)
    cpt_rel = torch.tensor(batch['cpt_rel']).to(device)
    length = batch['length']

    # Editor output
    pitch_changes = torch.tensor(lay_flat(batch['pitch_changes']))
    n_inserts = torch.tensor(lay_flat(batch['n_inserts']))
    
    return chd, [atr, cpt_rel, length], pitch_changes, n_inserts

def lay_flat(lst):
    # Pool sublist elements into a big list
    out = []
    for line in lst:
        for ele in line:
            out.append(ele)
    return out