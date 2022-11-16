# POP909-PIANOROLL-SERIES

POP909-PIANOROLL-SERIES is a quantized version of the original POP909 dataset. The original MIDI file and their corresponding `beat_midi.txt` and `chord_midi.txt` annotation files are processed into a piano-roll representation with different quantization rate such as 16th note, triplets, 32th notes etc.



## Sub-folders (or zipped files)

Each sub folder (or zipped file) is the piano-roll representation under a certain quantization rate. 

* Original: no quantization and any other transformation is applied.
* Raw-beat-without-quantization: the note onsets and offsets are transformed from second unit into beat unit, and shifted according to beat positions. However, no quantization is applied.
* n-bin-quantization: divide a beat into n bins and apply quantization to the transformed and shifted note onsets and offsets.



## `.Npz` file in each sub folders

In each `.npz` file, there are 5 matrices under 5 keys: 'melody', 'bridge', 'piano', 'beat', 'chord'. 

* melody: a note matrix of the MELODY track (N * 4). Each row represents a note. The columns are for onset_in_beat, off_set_in_beat, MIDI_pitch, and velocity.
* bridge: a note matrix of the BRIDGE track (N * 4).
* piano: a note matrix of the PIANO track (N * 4).
* beat: a beat-look-up table (M * 6). **The n-th row records the n-th beat of the song.** TBD.
* chord: a chord-look-up table (M * 14). **The n-th row records the chord on the n-th beat of the song.** TBD.