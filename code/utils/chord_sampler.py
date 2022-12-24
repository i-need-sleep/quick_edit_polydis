import torch
import random 

def decimalToBinary(n):
    return bin(n).replace("0b", "")

def gen_chords(step_func):
    out = torch.zeros(8, 36)
    for i in range(out.shape[0]):
        out[i] = step_func()
    return out

# Generate a random chord
# Uniformly sample a triad in root position.
def random_triad_step():
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

def random_any_step():
    # Randomly draw a root / bass
    root = torch.randint(high=12, size=(1, ))
    bass = torch.randint(high=12, size=(1, )) # Remember this is relative to the root
    
    out = torch.zeros(1, 36)
    out[0, root] = 1
    out[0, bass + 24] = 1

    # Uniformly randomly assign the chroma
    while True:
        out[0, 12: 24] = 0
        uni = random.randint(0, 2 ** 12 - 1)
        bin = decimalToBinary(uni).ljust(12, '0')
        for i, b in enumerate(bin):
            out[0, 12 + i] = int(b)
        
        # The root / bass must be in the chroma
        # Otherwise, do rejection sampling
        if out[0, 12 + root] == 1 and out[0, (root + bass) % 12 + 12] == 1:
            break

    return out

# Sample chords according to their frequency in Pop909
class Sampler909():
    def __init__(self, dataset):
        self.dataset = dataset

    def draw_chord(self):
        prog = self.draw_prog()
        idx = random.randint(0, 7)
        return torch.tensor(prog[idx, :]).float()
    
    def draw_prog(self):
        idx = random.randint(0, len(self.dataset))
        _, _, _, _, prog = self.dataset[idx]
        return torch.tensor(prog).float()
    

if __name__ == '__main__':
    out = random_any_step()
    print(out[0, :12])
    print(out[0, 12: 24])
    print(out[0, 24: ])