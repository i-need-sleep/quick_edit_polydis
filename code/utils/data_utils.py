import pretty_midi
import torch

def prep_batch(batch, device, include_original_notes=False, swap_original_rules=False, altered_atr_original_rel=False):

    chd = batch['chords'].to(device)

    # MuseBERT editor intput
    atr = torch.tensor(batch['atr']).to(device)
    cpt_rel = torch.tensor(batch['cpt_rel']).to(device)
    length = batch['length']

    # Editor output
    pitch_changes = torch.tensor(lay_flat(batch['pitch_changes'])).to(device)
    n_inserts = torch.tensor(lay_flat(batch['n_inserts'])).to(device)

    # Decoder
    decoder_atr_in = torch.tensor(batch['cpt_atr_dec']).to(device)
    decoder_rel_in = torch.tensor(batch['cpt_rel_dec']).to(device)
    decoder_atr_out = torch.tensor(batch['atr_dec']).to(device)
    decoder_len = batch['length_dec']
    decoder_output_mask = batch['output_mask_dec'].to(device)
    
    decoder_atr_out = decoder_atr_out[decoder_output_mask > 0]

    atr_original = torch.tensor(batch['atr_original']).to(device)
    rel_original = torch.tensor(batch['rel_original']).to(device)

    if altered_atr_original_rel:
        chd, [atr, rel_original, length], pitch_changes, n_inserts, [decoder_atr_in, decoder_rel_in, decoder_len], decoder_atr_out, decoder_output_mask

    if include_original_notes:

        if not swap_original_rules:
            return chd, [atr, cpt_rel, length, atr_original], pitch_changes, n_inserts, [decoder_atr_in, decoder_rel_in, decoder_len], decoder_atr_out, decoder_output_mask
        else:
            return chd, [atr_original, rel_original, length, atr], pitch_changes, n_inserts, [decoder_atr_in, decoder_rel_in, decoder_len], decoder_atr_out, decoder_output_mask
    
    return chd, [atr, cpt_rel, length], pitch_changes, n_inserts, [decoder_atr_in, decoder_rel_in, decoder_len], decoder_atr_out, decoder_output_mask

def prep_batch_inference(batch, device, ref=True, include_original_notes=False, swap_original_rules=False):

    chd = batch['chords'].to(device)

    # MuseBERT editor intput
    atr = torch.tensor(batch['atr']).to(device)
    cpt_rel = torch.tensor(batch['cpt_rel']).to(device)
    length = batch['length']

    notes_ref = []
    if ref:
        notes_ref = batch['notes_ref']

    if include_original_notes:
        atr_original = torch.tensor(batch['atr_original']).to(device)

        if not swap_original_rules: 
            return chd, [atr, cpt_rel, length, atr_original], notes_ref
        else:
            rel_original = torch.tensor(batch['rel_original']).to(device)
            return chd, [atr_original, rel_original, length, atr], notes_ref


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

def onset_pitch_duration_prettymidi_notes(nmat, step_size = 0.25):
    out = []
    for n in nmat:
        note = pretty_midi.Note(
            start = n[0] * step_size,
            end = (n[0] + n[2]) * step_size,
            pitch = n[1],
            velocity = 100
            )
        out.append(note)
    return out