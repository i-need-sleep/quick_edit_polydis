import dataset as dtst
from model import DisentangleVAE
import fun_utils
import torch
from torch.nn.utils.rnn import pack_padded_sequence
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
    model.to(device)

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
                        pr_mat = pr_mat[0].to(device)
                        chord = fun_utils.gen_chord().unsqueeze(0).to(device)

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
                                    # Spell out decider
                                    z, inference, x, lengths, teacher_forcing_ratio1, teacher_forcing_ratio2 = dec_z, True, None, None, 0., 0.
                                    # z: (B, z_size)
                                    # x: (B, num_step, max_simu_note, note_emb_size)
                                    batch_size = z.size(0)
                                    z_hid = model.decoder.z2dec_hid_linear(z).unsqueeze(0)
                                    # z_hid: (1, B, dec_time_hid_size)
                                    z_in = model.decoder.z2dec_in_linear(z).unsqueeze(1)
                                    # z_in: (B, dec_z_in_size)

                                    if inference:
                                        assert x is None
                                        assert lengths is None
                                        assert teacher_forcing_ratio1 == 0
                                        assert teacher_forcing_ratio2 == 0
                                    else:
                                        x_summarized = x.view(-1, model.decoder.max_simu_note, model.decoder.note_emb_size)
                                        x_summarized = pack_padded_sequence(x_summarized, lengths.view(-1),
                                                                            batch_first=True,
                                                                            enforce_sorted=False)
                                        x_summarized = model.decoder.dec_notes_emb_gru(x_summarized)[-1].\
                                            transpose(0, 1).contiguous()
                                        x_summarized = x_summarized.view(-1, model.decoder.num_step,
                                                                        2 * model.decoder.dec_emb_hid_size)

                                    pitch_outs = []
                                    dur_outs = []
                                    token = model.decoder.dec_init_input.repeat(batch_size, 1).unsqueeze(1)
                                    # (B, 2 * dec_emb_hid_size)

                                    for t in range(model.decoder.num_step):
                                        with record_function('decode_step'):
                                            notes_summary, z_hid = \
                                                model.decoder.dec_time_gru(torch.cat([token, z_in], dim=-1), z_hid)
                                            if inference:
                                                pitch_out, dur_out, predicted_notes, predicted_lengths = \
                                                    model.decoder.decode_notes(notes_summary, batch_size, None,
                                                                    inference, teacher_forcing_ratio2)
                                            else:
                                                pitch_out, dur_out, predicted_notes, predicted_lengths = \
                                                    model.decoder.decode_notes(notes_summary, batch_size, x[:, t],
                                                                    inference, teacher_forcing_ratio2)
                                            pitch_outs.append(pitch_out.unsqueeze(1))
                                            dur_outs.append(dur_out.unsqueeze(1))
                                            if t == model.decoder.num_step - 1:
                                                break

                                            teacher_force = random.random() < teacher_forcing_ratio1
                                            if teacher_force and not inference:
                                                token = x_summarized[:, t].unsqueeze(1)
                                            else:
                                                token = pack_padded_sequence(predicted_notes,
                                                                            predicted_lengths.cpu(),
                                                                            batch_first=True,
                                                                            enforce_sorted=False)
                                                token = model.decoder.dec_notes_emb_gru(token)[-1].\
                                                    transpose(0, 1).contiguous()
                                                token = token.view(-1, 2 * model.decoder.dec_emb_hid_size).unsqueeze(1)
                                    pitch_outs = torch.cat(pitch_outs, dim=1)
                                    dur_outs = torch.cat(dur_outs, dim=1)
                            
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