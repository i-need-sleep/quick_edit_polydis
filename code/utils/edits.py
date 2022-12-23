import random

class DefaultEditSet():
    '''
    Output format:
    MuseBERT input: [[start, pitch, duration],[...],...]
    Edit operations: change pitch to (0, 128], 0 = delete
    Insert operations: # tokens to add at each time step
    '''
    def __init__(self):
        self.step_size = 0.25
        self.n_steps = 32
        self.delete_idx = 0 # pitch for delete
        self.pitch_range = 128 # counting from 0, not including the cap
    
    def get_edits(self, notes_in, notes_ref):
        notes_out = [] # [[start, pitch, duration], ...]
        pitch_changes = [] # [[target_probs]], ...] aligned with notes_out. target_probs has len = self.pitch_range 
        n_inserts = [0 for _ in range(self.n_steps)] # [num_of_inserts_at_step, ...]
        inserts = [[] for _ in range(self.n_steps)] # [[pitch, duration], ...] for each step...
        # ...Might include some pitches in pitch changes. 
        # REMEMBER TO ADJUST THE ORACLE OUT AT TRAINING TIME
        decoder_notes_in = []

        for step in range(self.n_steps):
            note_groups_in = self._find_note_groups(notes_in, step)
            note_groups_ref = self._find_note_groups(notes_ref, step)

            notes_out_step, pitch_changes_step, n_inserts_step, inserts_step, decoder_notes_in_step = self._get_edits_step(note_groups_in, note_groups_ref, step)

            notes_out += notes_out_step
            pitch_changes += pitch_changes_step
            n_inserts[step] = n_inserts_step
            inserts[step] = inserts_step
            decoder_notes_in += decoder_notes_in_step
            
        return notes_out, pitch_changes, n_inserts, inserts, decoder_notes_in

    def _get_edits_step(self, note_groups_in, note_groups_ref, start):
        note_out = []
        pitch_changes = []
        n_inserts = 0
        inserts = []

        decoder_notes_in = [] # Context for the decoder [[start, pitch, duration], ...]
        
        for dur, pitches in note_groups_in.items():

            if dur not in note_groups_ref.keys():
                for pitch in pitches:
                    note_out.append([start, pitch, dur])
                    p_change = [0 for _ in range(self.pitch_range)]
                    p_change[self.delete_idx] = 1 
                    pitch_changes.append(p_change)

            else:
                # Build the target_prob list
                p_change = [0 for _ in range(self.pitch_range)]
                pitches_ref = note_groups_ref[dur]
                for pitch_ref in pitches_ref:
                    p_change[pitch_ref] += 1

                # Deletes
                if len(pitches_ref) < len(pitches):
                    p_change[self.delete_idx] = len(pitches) - len(pitches_ref)
                
                # Normalize
                s = sum(p_change)
                for idx, p in enumerate(p_change):
                    p_change[idx] = p / s

                # Build notes_out, pitch_changes
                for pitch in pitches:
                    note_out.append([start, pitch, dur])
                    pitch_changes.append(p_change)

                # Build inserts. Include all ref pitches with the same onset/dur. Adjust the oracle during training. 
                if len(pitches_ref) > len(pitches):
                    n_inserts += len(pitches_ref) - len(pitches)
                    for p in pitches_ref:
                        inserts.append([p, dur])

                # Build decoder note inputs
                selected_pitches = random.sample(pitches_ref, min([len(pitches), len(pitches_ref)]))
                for p in selected_pitches:
                    decoder_notes_in.append([start, p, dur])
                    

        
        # Handle "pure" inserts
        for dur, pitches in note_groups_ref.items():
            if dur not in note_groups_in.keys():
                n_inserts += len(pitches)
                for p in pitches:
                    inserts.append([p, dur])

        return note_out, pitch_changes, n_inserts, inserts, decoder_notes_in

    def _find_note_groups(self, notes, start):
        # Find note groups with the same onset/durations,
        # Output {duration: [pitches, ...], ...}
        out = {}
        for note in notes:
            if note.start / self.step_size == start:
                dur = int((note.end - note.start) / self.step_size)
                pitch = note.pitch
                if dur not in out.keys():
                    out[dur] = [pitch]
                else:
                    out[dur].append(pitch)
        return out

    def prep_decoder_notes(self, inserts_line, decoder_notes_in_line):
        print(inserts_line)
        print(decoder_notes_in_line)
        exit()

        return 