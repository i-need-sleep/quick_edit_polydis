import dataset as dtst
from model import DisentangleVAE
import fun_utils
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from amc_dl.torch_plus.train_utils import get_zs_from_dists, kl_with_normal
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def profile_polydis(n_iter=3):
    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    model = DisentangleVAE.init_model(device)
    model_path = 'result/models/disvae-nozoth_epoch.pt'  
    model.load_model(model_path, map_location=device)

    # Load the dataset 
    shift_low = -6
    shift_high = 6
    num_bar = 2
    contain_chord = True
    fns = dtst.collect_data_fns()
    dataset = dtst.wrap_dataset(fns, np.arange(len(fns)), shift_low, shift_high,
                                num_bar=num_bar, contain_chord=contain_chord)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # Set up the profiler
    activities = activities=[ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("big loop"):
            iter = 0
            while True:
                for batch in loader:
                    iter += 1
                    if iter > n_iter:
                        break

                    with record_function("batch"):
                        melody, pr, pr_mat, ptree, _ = batch
                        pr_mat = pr_mat[0]
                        chord = fun_utils.gen_chord().unsqueeze(0)

                        with record_function("Polydis inference"):
                            # Spell out the swap function and profile each component
                            model.eval()
                            with torch.no_grad():
                                with record_function('chord encoder'):
                                    dist_chd = model.chd_encoder(chord.float())
                                
                                with record_function('rhy encoder'):
                                    dist_rhy = model.rhy_encoder(pr_mat.float())

                                with record_function('sample z'):
                                    z_chd, z_rhy = get_zs_from_dists([dist_chd, dist_rhy], False)
                                    dec_z = torch.cat([z_chd, z_rhy], dim=-1)
                                
                                with record_function('decode'):
                                    pitch_outs, dur_outs = model.decoder(dec_z, True, None,
                                                                        None, 0., 0.)
                            
                                with record_function('process output'):
                                    polydis_out, _, _ = model.decoder.output_to_numpy(pitch_outs, dur_outs)

                        with record_function("decode output to notes"):
                            _, notes_polydis = model.decoder.grid_to_pr_and_notes(polydis_out.squeeze(0).astype(int))
                
                if iter > n_iter:
                    break

    # Print/Save results
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    prof.export_chrome_trace(f"trace_{device}_{n_iter}iter.json")
    return

if __name__ == '__main__':
    profile_polydis()