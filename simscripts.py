#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 20:32:28 2025

@author: franp
"""

import pyroomacoustics as pra
import numpy as np
import json
import copy

def mic_array(mic_amount, mic_start, mic_dist):
    """
    Generates a linear microphone array along the x-axis.

    Parameters
    ----------
    mic_amount : int
        Amount of mics in array.
    mic_start : list or tuple of float
        Starting position [x, y, z] of the first microphone.
    mic_dist : float
        Distance between adjacent microphones in meters.
    n_mics : int, optional
        Number of microphones in the array. Default is 4.

    Returns
    -------
    np.ndarray
        Microphone positions as a 2D array of shape (3, n_mics),
        where the first row is x, the second is y, and the third is z.
    """
    
    mic_pos = np.array([
        [mic_start[0] + i * mic_dist, mic_start[1], mic_start[2]] for i in range(mic_amount)
    ]).T
    return mic_pos

def room_sim(room_dim, rt60, mic_pos, source_pos, signal, fs=44100):
    """
    Simulates sound propagation in a shoebox-shaped room using the image source method (ISM).

    Parameters
    ----------
    room_dim : list of float
        Dimensions of the room [length, width, height] in meters.
    rt60 : float
        Desired reverberation time in seconds.
    mic_pos : np.ndarray
        Microphone positions as a 2D array of shape (3, n_mics),
        where the first row is x, the second is y, and the third is z.
    source_pos : list of float
        Position [x, y, z] of the sound source.
    signal : np.ndarray
        Audio signal to be played by the source (1D array).
    fs : int, optional
        Sampling frequency in Hz. Default is 44100.

    Returns
    -------
    room : np.ndarray
        Simulated microphone signals as a 2D array with shape (n_mics, n_samples).

    """
    # Se agregan los micrófonos restantes cada mic_dist
    
    
    e_absorption, max_order = pra.inverse_sabine(rt60+1e-12, room_dim)
    
    # Se crea sala y se agrega fuente
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order)
    room.add_source(source_pos, signal=signal)
    
    # Se agrega arreglo de micrófonos
    room.add_microphone_array(pra.MicrophoneArray(mic_pos, room.fs))

    room.simulate()

    return room




import copy
import json
import numpy as np

def expand_param(dicc_base, param_name, step, n=50, filename="config"):
    """
    Expands a scalar or 3D parameter in a dictionary into `n` values via step or range.

    Parameters
    ----------
    dicc_base : dict
        Original simulation dictionary.
    param_name : str
        Name of the parameter to vary.
    step : float, list or tuple
        - Scalar step (e.g., 0.1)
        - Scalar range: [start, stop]
        - Vector step: [dx, dy, dz]
        - Vector range: [[x0, y0, z0], [x1, y1, z1]]
    n : int
        Number of values to generate.
    filename : str
        Output JSON filename (no extension).

    Returns
    -------
    dict
        New dictionary with parameter expanded.
    """
    dicc_new = copy.deepcopy(dicc_base)

    if param_name not in dicc_new:
        raise KeyError(f"'{param_name}' not found in dictionary")

    val = dicc_new[param_name]

    # ESCALAR
    if isinstance(val, (int, float)):
        if isinstance(step, (int, float)):
            values = [val + k * step for k in range(n)]
        elif isinstance(step, (list, tuple)) and len(step) == 2:
            values = list(np.linspace(step[0], step[1], n))
        else:
            raise ValueError("For scalar parameters, step must be a number or a [start, stop] list.")
        dicc_new[param_name] = [round(float(v), 6) for v in values]

    # VECTOR 3D
    elif isinstance(val, (list, tuple)) and len(val) == 3:
        if isinstance(step, (list, tuple)) and len(step) == 3 and all(isinstance(x, (int, float)) for x in step):
            # step vector
            dicc_new[param_name] = [
                [val[i] + k * step[i] for i in range(3)] for k in range(n)
            ]
        elif isinstance(step, (list, tuple)) and len(step) == 2 and all(len(p) == 3 for p in step):
            # range vector
            start = np.array(step[0], dtype=float)
            end = np.array(step[1], dtype=float)
            path = [list(start + (end - start) * k / (n - 1)) for k in range(n)]
            dicc_new[param_name] = [[round(x, 6) for x in point] for point in path]
        else:
            raise ValueError("For 3D vectors, step must be [dx,dy,dz] or [[x0,y0,z0],[x1,y1,z1]]")
    else:
        raise TypeError(f"Unsupported type for parameter '{param_name}': {type(val)}")

    with open(f"{filename}.json", 'w') as f:
        json.dump(dicc_new, f, indent=4)

    return dicc_new




def simulate_from_config(signal, config_file):
    """
    Simulates a set of rooms based on a configuration JSON file and returns the mic signals for each.

    Parameters
    ----------
    signal : np.ndarray
        Audio signal to be used as the source.
    config_file : str
        Path to the configuration JSON file.

    Returns
    -------
    all_signals : list of np.ndarray
        List of simulated mic signals for each variation.
    """
    with open(config_file, "r") as f:
        config = json.load(f)

    # Detect which parameter was expanded (i.e., which one has multiple values)
    expanded_param = None
    for key, value in config.items():
        if isinstance(value, list):
            # Ignore 3D vector values (e.g., [x, y, z])
            if len(value) > 0 and isinstance(value[0], (int, float)) and len(value) > 3:
                expanded_param = key
                break
            elif len(value) > 0 and isinstance(value[0], list):  # list of vectors (e.g., source_pos variations)
                expanded_param = key
                break

    if expanded_param is None:
        raise ValueError("No expanded parameter found in config.")

    # Determine number of variations
    num_variations = len(config[expanded_param])

    all_signals = []

    for i in range(num_variations):
        # Prepare configuration for current iteration
        current_config = {
            key: (value[i] if key == expanded_param else value)
            for key, value in config.items()
        }

        mic_pos = mic_array(
            current_config["mic_amount"],
            current_config["mic_start"],
            current_config["mic_dist"]
        )

        room = room_sim(
            current_config["room_dim"],
            current_config["rt60"],
            mic_pos,
            current_config["source_pos"],
            signal,
            current_config["fs"]
        )

        all_signals.append(room.mic_array.signals)

    return all_signals



dicc_base = {
    "room_dim": [10, 10, 10], 
    "rt60": 0.5,
    "mic_amount": 4,
    "mic_start": [1, 1, 1],
    "mic_dist": 0.1,
    "source_pos": [5, 5, 1],
    "fs": 44100}
    


