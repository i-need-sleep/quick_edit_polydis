import torch
import random 

# Generate a random chord
# For now, uniformly sample a triad in root position. At actual training time, consider basing the sample distribution on noised frequencies. 
def gen_chord():
    out = torch.zeros(8, 36)
    for i in range(out.shape[0]):
        out[i] = gen_chord_step()
    return out

def gen_chord_step():
    root = torch.randint(high=12, size=(1, ))
    bass = 0
    
    out = torch.zeros(1, 36)
    out[0, root] = 1
    out[0, bass + 24] = 1
    
    out[0, 12 + root] = 1
    out[0, 12 + (root + 7) % 12] = 1

    if random.random() < 0.5:
        # Minor
        out[0, 12 + (root + 4) % 12] = 1
    else:
        out[0, 12 + (root + 3) % 12] = 1

    return out