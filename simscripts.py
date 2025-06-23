#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 20:32:28 2025

@author: franp
"""

import pyroomacoustics as pra
import numpy as np

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



