import numpy as np
from dtw import dtw

def align_handwriting_to_speech(hw_seq, sp_seq):
    dist = lambda x, y: np.linalg.norm(x - y)

    alignment = dtw(hw_seq, sp_seq, dist=dist)

    path = np.array(alignment.index1), np.array(alignment.index2)

    aligned_hw_seq = []

    for j in range(len(sp_seq)):
        matched_hw_indices = [i for i, jj in zip(*path) if jj == j]
        if matched_hw_indices:
            mean_vec = np.mean(hw_seq[matched_hw_indices], axis=0)
        else:
            mean_vec = np.zeros(hw_seq.shape[1])
        aligned_hw_seq.append(mean_vec)

    aligned_hw_seq = np.vstack(aligned_hw_seq)

    return aligned_hw_seq
