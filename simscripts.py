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




def expand_param(dicc_base, param_name, step, n=50, filename="config"):
    """
    Expands a scalar or 3D coordinate parameter in a base dictionary, generating 
    a list of `n` values based on a specified step. Automatically saves the result as a JSON file.

    Parameters
    ----------
    dicc_base : dict
        Base dictionary with the original parameters.
    param_name : str
        Name of the parameter to expand.
    step : float or list/tuple of 3 elements
        Step size. Must be:
            - Scalar if the original value is scalar.
            - List or tuple of length 3 if the original value is a 3D coordinate.
    n : int, optional
        Number of values to generate. Default is 50.
    filename : str, optional
        Base name for the JSON file (without extension). Default is 'config'.

    Returns
    -------
    dict
        A copy of the dictionary with the expanded parameter.

    Raises
    ------
    KeyError
        If the parameter name does not exist in the dictionary.
    ValueError
        If step is not valid for the parameter type.
    TypeError
        If the parameter type is unsupported.
    """
    dicc_new = copy.deepcopy(dicc_base)

    if param_name not in dicc_new:
        raise KeyError(f"Parameter '{param_name}' not found in the dictionary.")

    val = dicc_new[param_name]

    # 3D coordinate case
    if isinstance(val, (list, tuple)) and len(val) == 3:
        if not (isinstance(step, (list, tuple)) and len(step) == 3):
            raise ValueError("For 3D vector parameters, step must also be a list or tuple of length 3.")
        dicc_new[param_name] = [
            [val[i] + k * step[i] for i in range(3)] for k in range(n)
        ]
    # Scalar case
    elif isinstance(val, (int, float)):
        if not isinstance(step, (int, float)):
            raise ValueError("For scalar parameters, step must be a number (int or float).")
        dicc_new[param_name] = [val + k * step for k in range(n)]
    else:
        raise TypeError(f"Unsupported parameter type for '{param_name}': {type(val)}")

    # Save dictionary to JSON file
    with open(f"{filename}.json", 'w') as f:
        json.dump(dicc_new, f, indent=4)

    return dicc_new



dicc_base = {"rt60" : 0.5, "mic_amount" : 4, "mic_start" : [1, 1, 1], "mic_dist" : 0.1, "source_pos" : [5, 5, 1], "fs" : 44100}


