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

    # Decoder
    decoder_atr_in = torch.tensor(batch['cpt_atr_dec'])
    decoder_rel_in = torch.tensor(batch['cpt_rel_dec'])
    decoder_atr_out = torch.tensor(batch['atr_dec'])
    decoder_len = batch['length_dec']
    decoder_output_mask = batch['output_mask_dec']
    
    decoder_atr_out = decoder_atr_out[decoder_output_mask > 0]
    
    return chd, [atr, cpt_rel, length], pitch_changes, n_inserts, [decoder_atr_in, decoder_rel_in, decoder_len], decoder_atr_out, decoder_output_mask

def prep_batch_inference(batch, device, ref=True):

    chd = batch['chords'].to(device)

    # MuseBERT editor intput
    atr = torch.tensor(batch['atr']).to(device)
    cpt_rel = torch.tensor(batch['cpt_rel']).to(device)
    length = batch['length']

    notes_ref = []
    if ref:
        notes_ref = lay_flat(batch['notes_ref'])
    
    return chd, [atr, cpt_rel, length], notes_ref

def lay_flat(lst):
    # Pool sublist elements into a big list
    out = []
    for line in lst:
        for ele in line:
            out.append(ele)
    return out

def prettymidi_notes_to_onset_pitch_duration(notes, step_size = 0.25):
    out = []
    for note in notes:
        out.append([int(note.start / step_size), note.pitch, int((note.end - note.start) / step_size)])
    return out