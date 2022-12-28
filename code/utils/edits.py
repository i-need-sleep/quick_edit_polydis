import random

import torch
import networkx as nx

import utils.data_utils

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
        out = []
        for start, inserts in enumerate(inserts_line):
            for [pitch, dur] in inserts:
                note = [start, pitch, dur]
                if note not in decoder_notes_in_line:
                    out.append(note)

        return out

    def _find_indices(self, notes, i):
        # Given a nmat, find the indices of all notes with the same onset/duration as notes[i] (including i)
        onset = notes[i][0]
        dur = notes[i][2]
        out = []
        for idx, note in enumerate(notes):
            if note[0] == onset and note[2] == dur:
                out.append(idx)
        return out
    
    def edits_to_nmat(self, nmat_line, edits_line):
        # Since we predict pitch changes with soft labels, we have to do a softmax over all note groups with the same onset/dur
        # Do this note-by-note.
        notes_context_line = []
        processed_inds = []
        for idx in range(len(nmat_line)):
            if idx not in processed_inds:
                inds = self._find_indices(nmat_line, idx)
                processed_inds += inds
                cur_logits = edits_line[inds]
                cur_probs = torch.nn.Softmax(dim=0)(cur_logits)
                for i, ind in enumerate(inds):
                    # Find the highest prob
                    highest_idx = torch.max(cur_probs, dim=1).indices[i].item()

                    # Add the new pitch-changed note
                    if highest_idx > 0:
                        notes_context_line.append([nmat_line[ind][0], highest_idx, nmat_line[ind][2]])

                    # Adjust the probs
                    cur_probs[:, highest_idx] -= 1 / len(inds)
        return notes_context_line

class MFMCEditSet():
    def __init__(self):
        self.step_size = 0.25
        self.n_steps = 32

        # Edit operations
        self.onset_ops = [i for i in range(-1, 2)]
        self.pitch_ops = [i for i in range(-3, 4)]
        self.dur_ops = [i for i in range(-3, 4)]
        self.delete_idx = len(self.onset_ops) * len(self.pitch_ops) * len(self.dur_ops)

        # Edit costs
        self.onset_costs = {}
        for op in self.onset_ops:
            self.onset_costs[op] = abs(op)

        self.pitch_costs = {}
        for op in self.pitch_ops:
            self.pitch_costs[op] = abs(op)

        self.dur_costs = {}
        for op in self.dur_ops:
            self.dur_costs[op] = abs(op)

        self.delete_cost = 100
        self.out_of_range_cost = 999

        # n_editing operations
        self.pitch_range = self.delete_idx + 1

    def get_edits(self, notes_in, notes_ref):

        # notes_out = [] # [[start, pitch, duration], ...]
        # edits = [] # [[target_probs]], ...] aligned with notes_out. This should be a hard label, but we output probs so that this is exchangable with the default edit set
        # n_inserts = [0 for _ in range(self.n_steps)] # [num_of_inserts_at_step, ...]
        # inserts = [[] for _ in range(self.n_steps)] # [[pitch, duration], ...] for each step
        # decoder_notes_in = [] # [[start, pitch, duration], ...]

        # Build atr input for MuseBERT
        notes_out = utils.data_utils.prettymidi_notes_to_onset_pitch_duration(notes_in, step_size=self.step_size)
        notes_ref = utils.data_utils.prettymidi_notes_to_onset_pitch_duration(notes_ref, step_size=self.step_size)

        # Solve the max flow min cost problem to get edits
        # Build the graph
        g = self._build_di_graph(notes_out, notes_ref)
        # Solve the graph
        flow = nx.max_flow_min_cost(g, -1, -2)
        # Build Edits
        edits, n_inserts, inserts, decoder_notes_in = self._flow_to_edits(notes_out, notes_ref, flow)
        
        return  notes_out, edits, n_inserts, inserts, decoder_notes_in

    def _flow_to_edits(self, notes_out, notes_ref, flow):

        edits = [] 
        n_inserts = [0 for _ in range(self.n_steps)] # [num_of_inserts_at_step, ...]
        inserts = [[] for _ in range(self.n_steps)] # [[pitch, duration], ...] for each step
        decoder_notes_in = [] # [[start, pitch, duration], ...]

        for out_idx, note_out in enumerate(notes_out):

            edit = [0 for _ in range(self.delete_idx + 1)]

            # Find the where this node flows to
            for key, val in flow[out_idx].items():
                if val == 1:
                    dest_idx = key
                    break
            
            # Register deletion
            if dest_idx == -3:
                edit[self.delete_idx] = 1.
                edits.append(edit)
                continue
            
            # Register onset/pitch/dur changes
            dest_note = notes_ref[dest_idx - len(notes_out)]

            onset_diff = dest_note[0] - note_out[0]
            pitch_diff = dest_note[1] - note_out[1]
            dur_diff = dest_note[2] - note_out[2]

            onset_idx = self.onset_ops.index(onset_diff)
            pitch_idx = self.pitch_ops.index(pitch_diff)
            dur_idx = self.dur_ops.index(dur_diff)

            # De-factorise into an indice
            edit_idx = onset_idx * (len(self.pitch_ops) * len(self.dur_ops)) + pitch_idx * len(self.dur_ops) + dur_idx
            edit[edit_idx] = 1.
            edits.append(edit)
            decoder_notes_in.append(dest_note)

        # Handle insertion
        for note_ref in notes_ref:
            onset, pitch, dur = note_ref
            edited = False
            for note in decoder_notes_in:
                if note[0] == onset and note[1] == pitch and note[2] == dur:
                    edited = True
                    break
            if not edited:
                inserts[onset].append([pitch, dur])
                n_inserts[onset] += 1
                
        return edits, n_inserts, inserts, decoder_notes_in
    
    def _build_di_graph(self, notes_out, notes_ref):
        '''
        Nodes idx:
        -1: source
        -2: target
        -3: delete
        [0, len(notes_out)): notes_out
        [len(note_out), len(notes_out) + len(notes_ref)): notes_ref
        '''
        g = nx.DiGraph()

        # Start from node -1. Add edges to notes_out.
        g.add_edges_from([
            (-1, i, {'capacity': 1, 'weight': 0}) for i in range(len(notes_out))
        ])

        # Edges from notes_ref to the target node
        g.add_edges_from([
            (i, -2, {'capacity': 1, 'weight': 0}) for i in range(len(notes_out), len(notes_out) + len(notes_ref))
        ])

        # From the delete node to the target node
        g.add_edges_from([
            (-3, -2, {'capacity': len(notes_out), 'weight': 0})
        ])

        # From notes_out to notes_ref and the delete node
        for out_idx, note_out in enumerate(notes_out):
            
            # delete node
            add_lst = [(out_idx, -3, {'capacity': 1, 'weight': self.delete_cost})]

            # notes_ref
            for ref_idx, note_ref in enumerate(notes_ref):
                cost = self._get_cost(note_out, note_ref)
                add_lst.append((out_idx, ref_idx + len(notes_out), {'capacity': 1, 'weight': cost}) )
            g.add_edges_from(add_lst)
            
        return g
    
    def _get_cost(self, note_out, note_ref):
        onset_diff = note_ref[0] - note_out[0]
        pitch_diff = note_ref[1] - note_out[1]
        dur_diff = note_ref[2] - note_out[2]

        if onset_diff in self.onset_costs.keys():
            cost = self.onset_costs[onset_diff]
        else:
            cost = self.out_of_range_cost

        if pitch_diff in self.pitch_costs.keys():
            cost += self.pitch_costs[pitch_diff]
        else:
            cost += self.out_of_range_cost

        if dur_diff in self.dur_costs.keys():
            cost += self.dur_costs[dur_diff]
        else:
            cost += self.out_of_range_cost

        return cost

    def prep_decoder_notes(self, inserts_line, decoder_notes_in_line):
        out = []
        for start, inserts in enumerate(inserts_line):
            for [pitch, dur] in inserts:
                note = [start, pitch, dur]
                if note not in decoder_notes_in_line:
                    out.append(note)

        return out

    def edits_to_nmat(self, nmat_line, edits_line):
        edits_line = torch.max(edits_line, dim=1).indices
        nmat_out = []

        for note_idx, note in enumerate(nmat_line):
            
            edit_idx = edits_line[note_idx]

            # Deletion
            if edit_idx == self.delete_idx:
                continue

            # Factorise edits
            onset, pitch, dur = note
            
            onset_idx = edit_idx // (len(self.pitch_ops) * len(self.dur_ops))
            pitch_idx = edit_idx % (len(self.pitch_ops) * len(self.dur_ops)) // len(self.dur_ops) 
            dur_idx = edit_idx % len(self.dur_ops)

            onset_diff = self.onset_ops[onset_idx]
            pitch_diff = self.pitch_ops[pitch_idx]
            dur_diff = self.dur_ops[dur_idx]

            note_out = [onset + onset_diff, pitch + pitch_diff, dur + dur_diff]

            # Fix illegel cases
            if note_out[0] < 0:
                note_out[0] = 0
            if note_out[0] > 31:
                note_out[0] = 31
            if note_out[1] < 2:
                note_out[1] = 2
            if note_out[1] > 127:
                note_out[1] = 127
            if note_out[2] < 1:
                note_out[2] = 1

            nmat_out.append(note_out)
            
        return nmat_out