# Nearest neighbours
# An naive rule set to apply HST. Consider using a learned one.
def nearest_neighbour(notes, chord_mat):
    notes_out = []
    for i in range(chord_mat.shape[0]):
        notes_out += apply_nn_step(notes, chord_mat[i: i+1, :], i)
    return notes_out

def apply_nn_step(notes, chord, step, quant_size=1):
    # Filter out notes starting in the window
    filtered_notes = []
    for note in notes:
        if note.start >= quant_size * step and note.start < quant_size * (step + 1):
            filtered_notes.append(note)
    
    for note in filtered_notes:
        pitch_dist = 999
        pit_out = 0
        for pit in range(20, 100):
            # Is this pitch included in the chroma?
            if chord[0, 12 + pit % 12] == 0:
                continue
            dist = abs(note.pitch - pit)
            if dist < pitch_dist:
                pitch_dist = dist
                pit_out = pit
        note.pitch = pit_out

    # Remove duplicates
    filtered_notes = remove_note_duplicates(filtered_notes)
    return filtered_notes

def remove_note_duplicates(notes):
    out = []
    for note in notes:
        add = True
        for note_out in out:
            if note.start//0.25 == note_out.start//0.25 and note.end//0.25 == note_out.end//0.25 and note.pitch == note_out.pitch:
                add = False
        if add:
            out.append(note)
    return out

# Identity transformation
def identity(notes, chord_mat):
    return notes
