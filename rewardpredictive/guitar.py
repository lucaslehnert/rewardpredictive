#
# Copyright (c) 2020 Lucas Lehnert <lucas_lehnert@brown.edu>
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#
from itertools import product

import numpy as np
import rlutils as rl

notes = ('A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#')


def get_fret_board_standard_tuning():
    fret_board = np.array([
        ['E', 'A', 'D', 'G', 'B', 'E'],
        ['F', 'A#', 'D#', 'G#', 'C', 'F'],
        ['F#', 'B', 'E', 'A', 'C#', 'F#'],
        ['G', 'C', 'F', 'A#', 'D', 'G'],
        ['G#', 'C#', 'F#', 'B', 'D#', 'G#'],
        ['A', 'D', 'G', 'C', 'E', 'A'],
        ['A#', 'D#', 'G#', 'C#', 'F', 'A#'],
        ['B', 'E', 'A', 'D', 'F#', 'B'],
        ['C', 'F', 'A#', 'D#', 'G', 'C'],
        ['C#', 'F#', 'B', 'E', 'G#', 'C#'],
        ['D', 'G', 'C', 'F', 'A', 'D'],
        ['D#', 'G#', 'C#', 'F#', 'A#', 'D#'],
        ['E', 'A', 'D', 'G', 'B', 'E'],
        ['F', 'A#', 'D#', 'G#', 'C', 'F'],
        ['F#', 'B', 'E', 'A', 'C#', 'F#'],
        ['G', 'C', 'F', 'A#', 'D', 'G'],
        ['G#', 'C#', 'F#', 'B', 'D#', 'G#'],
        ['A', 'D', 'G', 'C', 'E', 'A'],
        ['A#', 'D#', 'G#', 'C#', 'F', 'A#'],
        ['B', 'E', 'A', 'D', 'F#', 'B'],
        ['C', 'F', 'A#', 'D#', 'G', 'C'],
        ['C#', 'F#', 'B', 'E', 'G#', 'C#'],
        ['D', 'G', 'C', 'F', 'A', 'D'],
        ['D#', 'G#', 'C#', 'F#', 'A#', 'D#'],
        ['E', 'A', 'D', 'G', 'B', 'E']
    ], dtype=np.str)
    octave_idx = np.array([
        [0, 0, 1, 1, 1, 2],
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 1, 2, 2],
        [0, 1, 1, 1, 2, 2],
        [0, 1, 1, 1, 2, 2],
        [0, 1, 1, 2, 2, 2],
        [0, 1, 1, 2, 2, 2],
        [0, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 3],
        [1, 1, 1, 2, 2, 3],
        [1, 1, 2, 2, 2, 3],
        [1, 1, 2, 2, 2, 3],
        [1, 1, 2, 2, 2, 3],
        [1, 1, 2, 2, 3, 3],
        [1, 1, 2, 2, 3, 3],
        [1, 2, 2, 2, 3, 3],
        [1, 2, 2, 2, 3, 3],
        [1, 2, 2, 3, 3, 3],
        [1, 2, 2, 3, 3, 3],
        [1, 2, 2, 3, 3, 3],
        [2, 2, 2, 3, 3, 4],
        [2, 2, 2, 3, 3, 4],
        [2, 2, 3, 3, 3, 4],
        [2, 2, 3, 3, 3, 4],
        [2, 2, 3, 3, 3, 4]
    ], dtype=np.int)
    return fret_board, octave_idx


def guitar_melody_mdp(melody, fret_board=None, octave_idx=None, reward_max=1., reward_min=-1.):
    if fret_board is None:
        fret_board = get_fret_board_standard_tuning()[0]
    if octave_idx is None:
        octave_idx = get_fret_board_standard_tuning()[1]
    fret_board_1 = [''] + list(np.reshape(fret_board.transpose(), -1))
    octave_idx_1 = [-1] + list(np.reshape(octave_idx.transpose(), -1))
    melody_transitions = list(zip([''] + list(melody[:-1]), melody))

    num_s = len(fret_board_1)
    num_a = len(notes)
    t_mat = np.stack([np.eye(num_s, dtype=np.float32) for _ in range(num_a)])
    r_mat = np.ones([num_a, num_s, num_s], dtype=np.float32) * reward_min

    note_it = enumerate(notes)
    fret_it_1 = enumerate(zip(fret_board_1, octave_idx_1))
    fret_it_2 = enumerate(zip(fret_board_1, octave_idx_1))
    for (ni, na), (si, (n, o)), (sni, (nn, on)) in product(note_it, fret_it_1, fret_it_2):
        if (n, nn) in melody_transitions and na == nn and on in [1, 2, 3] and (o == -1 or o == on):
            r_mat[ni, si, sni] = reward_max
    for a, s in product(range(num_a), range(num_s)):
        rewarding_transitions = np.where(r_mat[a, s] == reward_max)[0]
        if len(rewarding_transitions) > 0:
            t_mat[a, s, s] = 0.
        for i in rewarding_transitions:
            t_mat[a, s, i] = 1.
    t_mat /= np.sum(t_mat, axis=-1, keepdims=True)  # normalize to correct transition probabilities out of start state.

    idx_goal_list = [i for i, (n, o) in enumerate(zip(fret_board_1, octave_idx_1)) if
                     n == melody[-1] and o in [1, 2, 3]]
    mdp = rl.environment.TabularMDP(t_mat, r_mat, idx_start_list=[0], idx_goal_list=idx_goal_list)
    return mdp


def phi_mat_from_fret_board(fret_board=None, octave_idx=None):
    if fret_board is None:
        fret_board = get_fret_board_standard_tuning()[0]
    if octave_idx is None:
        octave_idx = get_fret_board_standard_tuning()[1]
    num_frets, num_strings = np.shape(fret_board)
    num_notes = len(notes)
    phi_mat = np.zeros([num_frets * num_strings + 1, num_notes + 2], dtype=np.float32)
    fret_board_1 = np.reshape(fret_board.transpose(), -1)
    octave_idx_1 = np.reshape(octave_idx.transpose(), -1)

    phi_mat[0, 0] = 1.
    for i, (n, o) in enumerate(zip(fret_board_1, octave_idx_1)):
        if o == 0 or o == 4:
            phi_mat[i + 1, -1] = 1.
        else:
            j = notes.index(n)
            phi_mat[i + 1, j + 1] = 1.
    return phi_mat
