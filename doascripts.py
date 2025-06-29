#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 17:42:16 2025

@author: franp
"""

import numpy as np
from scipy.signal import correlate
from scipy.fft import fft, ifft
import json
import soundfile as sf
import copy
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import simscripts as sim
import generate as gen


def estimate_tdoa(sig1, sig2, fs):
    corr = correlate(sig1, sig2, mode='full')
    lag = np.argmax(corr) - (len(sig2) - 1)
    return lag / fs  # tiempo en segundos

def corr_norm(sig1, sig2, fs):
    corr = correlate(sig1, sig2, mode='full')
    lag = (np.argmax(corr) - (len(sig2) - 1))/fs
    return lag, corr/np.abs(np.max(corr))


def gcc(sig1, sig2, fs, method='classicfft', norm=False):
    """
    Compute the Generalized Cross-Correlation (GCC) function between two signals
    using a selected spectral weighting method.

    Parameters
    ----------
    sig1 : ndarray
        First input signal (1D array).
    sig2 : ndarray
        Second input signal (1D array).
    fs : int or float
        Sampling frequency in Hz.
    method : str, optional
        GCC weighting method. Options: 'phat', 'scot', 'roth', 'eckart', 'ml', 'classicfft' and 'classictemp'.
    norm : bool, optional
        Normalization of the CC results.

    Returns
    -------
    t_lags : ndarray
        Array of time lags in seconds.
    cc : ndarray
        GCC function evaluated over time lags.
    
    Raises
    ------
    ValueError
        If the method is not recognized.
    """
    n = len(sig1) + len(sig2) - 1

    if method == 'classictemp':
        cc = correlate(sig1, sig2, mode='full', method="direct")
        lags = np.arange(-len(sig2) + 1, len(sig1))
    else:
        SIG1 = fft(sig1, n=n)
        SIG2 = fft(sig2, n=n)
        G = SIG1 * np.conj(SIG2)

        P1 = np.abs(SIG1) ** 2
        P2 = np.abs(SIG2) ** 2

        if method == 'classicfft':
            W = 1.0
        elif method == 'phat':
            W = 1 / (np.abs(G) + 1e-12)
        elif method == 'scot':
            W = 1 / (np.sqrt(P1 * P2) + 1e-12)
        elif method == 'roth':
            W = 1 / (P2 + 1e-12)
        elif method == 'eckart':
            W = np.abs(G) / (P2 + 1e-12) ** 2
            W = W / (np.max(W) + 1e-12)
        elif method == 'ml':
            noise_power = np.mean(P2)
            W = np.abs(G) / (P2 + noise_power + 1e-12)
        else:
            raise ValueError("Invalid method. Choose from 'classictemp', 'classicfft', 'phat', 'scot', 'roth', 'eckart', 'ml'.")

        G_weighted = G * W
        cc = np.real(ifft(G_weighted))
        cc = np.fft.fftshift(cc)
        lags = np.arange(-(n // 2), (n + 1) // 2)

    t_lags = lags / fs
    if norm:
        cc = cc / np.max(np.abs(cc))
    return t_lags, cc


def true_doa(mic_pos, source_pos):
    """
    Computes the ground-truth Direction of Arrival (DOA) angle 
    from the first microphone to the source position.

    Parameters
    ----------
    mic_pos : np.ndarray
        Microphone positions as a 2D array of shape (3, N), 
        where rows are [x, y, z] and N is the number of microphones.
    source_pos : list or array-like
        Source position as a list or array of [x, y, z].

    Returns
    -------
    float
        Ground-truth DOA angle in degrees, measured in the XY plane
        from the positive x-axis towards the source direction.
        Range: [0, 360).
    """
    mic_origin = mic_pos[:, 0]  # usar primer micrófono como referencia
    src_vec = np.array(source_pos[:2]) - mic_origin[:2]  # vector desde mic a fuente

    angle_rad = np.arctan2(src_vec[1], src_vec[0])  # atan2(y, x)
    angle_deg = np.degrees(angle_rad) % 360

    return angle_deg



def gcc_tdoas(signals, fs, max_tau=None, method="phat", mic_pairs=None, c=343.0):
    """
    Compute TDOAs between microphone 0 and all others using GCC.

    Parameters
    ----------
    signals : np.ndarray
        Array of shape (n_mics, n_samples), containing microphone signals.
    fs : int or float
        Sampling frequency in Hz.
    max_tau : float, optional
        Maximum TDOA (in seconds) to consider. If None, uses full range.
    method : str, optional
        GCC method: "classicfft", "phat", "scot", "roth", "ml".
    mic_pairs : list of tuple, optional
        List of mic index pairs. If None, compares all to mic 0.
    c : float, optional
        Speed of sound (default: 343 m/s).

    Returns
    -------
    tdoas : list of float
        Estimated TDOA values (in seconds) for each mic pair.
    """
    n_mics, n_samples = signals.shape

    # Comparar todos contra el micrófono 0
    if mic_pairs is None:
        mic_pairs = [(0, i) for i in range(1, n_mics)]

    tdoas = []

    for i, j in mic_pairs:
        sig1 = signals[i]
        sig2 = signals[j]

        SIG1 = np.fft.fft(sig1)
        SIG2 = np.fft.fft(sig2)
        R = SIG1 * np.conj(SIG2)

        if method == "phat":
            R /= np.abs(R) + 1e-10
        elif method == "scot":
            R /= np.sqrt(np.abs(SIG1)**2 * np.abs(SIG2)**2 + 1e-10)
        elif method == "roth":
            R /= (np.abs(SIG2)**2 + 1e-10)
        elif method == "ml":
            Sxx = np.abs(SIG1)**2
            Syy = np.abs(SIG2)**2
            R /= (Sxx + Syy + 1e-10)
        elif method == "classicfft":
            pass
        else:
            raise ValueError(f"Método desconocido: {method}")

        corr = np.fft.ifft(R).real
        corr = np.fft.fftshift(corr)

        max_lag = n_samples // 2
        lags = np.arange(-max_lag, max_lag)

        if max_tau is not None:
            max_shift = int(fs * max_tau)
            center = len(corr) // 2
            corr = corr[center - max_shift : center + max_shift]
            lags = lags[center - max_shift : center + max_shift]

        peak_index = np.argmax(corr)
        tdoa = lags[peak_index] / fs
        tdoas.append(tdoa)

    return tdoas



def doa(tdoa, mic_positions, mic_pairs=None, c=343, return_all=False):
    """
    Calculates Direction of Arrival (DOA) angles from Time Differences of Arrival (TDOA) 
    using multiple microphone pairs (default: with respect to mic 0).

    Parameters
    ----------
    tdoa : array-like
        TDOAs in seconds, one per mic pair.
    mic_positions : np.ndarray
        Array of shape (3, n_mics), each column is [x, y, z] of one mic.
    mic_pairs : list of tuple of int, optional
        List of (i, j) mic index pairs corresponding to each specified tdoa.
        If None, uses (0, i) for i in range(1, n_mics).
    c : float, optional
        Speed of sound in m/s. Default is 343.
    return_all : bool, optional
        If True, return all DOAs. Else, return their average.

    Returns
    -------
    float or np.ndarray
        DOA in degrees, or array of DOAs if return_all is True.

    Notes
    -----
    Uses cos(theta) = tdoa * c / distance
    Angle is always in [0, 180] due to acos. You may resolve ambiguity separately.
    """
    tdoa = np.asarray(tdoa)
    mic_positions = np.asarray(mic_positions)

    if mic_positions.shape[0] != 3:
        raise ValueError("mic_positions must have shape (3, n_mics) — 3 rows for x, y, z.")

    n_mics = mic_positions.shape[1]

    # 🔄 Usar micrófono 0 como referencia si no se especifican pares
    if mic_pairs is None:
        mic_pairs = [(0, i) for i in range(1, n_mics)]

    if len(tdoa) != len(mic_pairs):
        raise ValueError("Number of TDOAs must match number of mic pairs.")

    doa_angles = []

    for tau, (i, j) in zip(tdoa, mic_pairs):
        d_vec = mic_positions[:, j] - mic_positions[:, i]
        d = np.linalg.norm(d_vec)
        cos_theta = np.clip(tau * c / d, -1, 1)
        angle = np.degrees(np.arccos(cos_theta))
        doa_angles.append(angle)

    doa_angles = np.array(doa_angles)

    return doa_angles if return_all else round(doa_angles.mean(), 2)


def batch_gcc_tdoas(all_signals, fs_list, mic_positions_list=None, mic_pairs=None,
                    method='classicfft', max_tau=None, c=343):
    """
    Computes TDOAs for a list of simulations, using mic 0 as reference if mic_pairs not given.

    Parameters
    ----------
    all_signals : list of np.ndarray
        List of (n_mics, n_samples) arrays, one per simulation.
    fs_list : list of int or float
        Sampling frequencies, one per simulation.
    mic_positions_list : list of np.ndarray, optional
        List of mic position arrays, needed to compute max_tau automatically if not given.
    mic_pairs : list of tuple of int, optional
        Microphone index pairs. If None, uses (0, i) for i=1..n_mics-1.
    method : str, optional
        GCC method. Default is 'classicfft'.
    max_tau : float, optional
        Maximum TDOA to consider (in seconds). If None, computed from mic geometry.
    c : float, optional
        Speed of sound in m/s. Default is 343.

    Returns
    -------
    list of list of float
        List of TDOAs per simulation.
    """
    all_tdoas = []

    for idx, (signals, fs) in enumerate(zip(all_signals, fs_list)):
        if max_tau is None:
            if mic_positions_list is None:
                raise ValueError("mic_positions_list is required to compute max_tau automatically")

            mic_pos = mic_positions_list[idx]
            n_mics = mic_pos.shape[1]

            if mic_pairs is None:
                mic_pairs_local = [(0, i) for i in range(1, n_mics)]
            else:
                mic_pairs_local = mic_pairs

            max_dist = max(np.linalg.norm(mic_pos[:, i] - mic_pos[:, j]) for i, j in mic_pairs_local)
            max_tau_sim = max_dist / c
        else:
            max_tau_sim = max_tau

        tdoas = gcc_tdoas(
            signals,
            fs,
            mic_pairs=mic_pairs if mic_pairs is not None else mic_pairs_local,
            method=method,
            max_tau=max_tau_sim
        )
        all_tdoas.append(tdoas)

    return all_tdoas


def batch_doas(all_tdoas, mic_positions_list, mic_pairs=None, c=343):
    """
    Computes DOA for each TDOA set and corresponding mic positions.

    Parameters
    ----------
    all_tdoas : list of list of float
        Each sublist contains TDOAs for a single simulation.
    mic_positions_list : list of np.ndarray
        Each element is a (3, n_mics) array of mic positions.
    mic_pairs : list of tuple, optional
        Mic index pairs. If None, uses (0, i) for i in 1..n_mics-1.
    c : float, optional
        Speed of sound.

    Returns
    -------
    list of float
        Average DOA (in degrees) per simulation.
    """
    doas_results = []

    for tdoas, mic_pos in zip(all_tdoas, mic_positions_list):
        n_mics = mic_pos.shape[1]

        # Usar micrófono 0 como referencia si no se especificaron pares
        pairs = mic_pairs if mic_pairs is not None else [(0, i) for i in range(1, n_mics)]

        doa_value = doa(tdoas, mic_pos, mic_pairs=pairs, c=c, return_all=False)
        doas_results.append(doa_value)

    return doas_results




def full_doa_pipeline(json_path, signal, mic_pairs=None, method='classicfft', max_tau=None, c=343, variable_param=None, return_error=True):
    """
    Loads configurations, simulates, calculates TDOAs and DOAs, with microphone noise optionally included from JSON.

    Parameters
    ----------
    json_path : str
        Path to the JSON file containing simulation parameters.
    signal : str or np.ndarray
        Source signal or path to .wav file.
    mic_pairs : list of tuple of int, optional
        Microphone pairs for TDOA/DOA.
    method : str, optional
        GCC method ('classic', 'phat', etc.).
    max_tau : float, optional
        Maximum expected TDOA.
    c : float, optional
        Speed of sound.
    variable_param : str, optional
        Name of the parameter being varied.
    return_error : bool, optional
        If True, returns absolute error compared to ground truth DOA.

    Returns
    -------
    tuple of (list, list, str)
        - Values of the varied parameter (x-axis).
        - Estimated DOAs or errors.
        - Name of the varied parameter.
    """
    # Leer el audio si es un path
    if isinstance(signal, str):
        signal_data, fs_signal = sf.read(signal)
        if signal_data.ndim > 1:
            signal_data = signal_data[:, 0]  # forzar mono
    else:
        signal_data = signal
        fs_signal = None

    # Leer JSON
    with open(json_path, 'r') as f:
        config_data = json.load(f)

    # Detectar parámetro variable
    if variable_param is not None:
        varied_param = variable_param
        if varied_param not in config_data:
            raise ValueError(f"'{varied_param}' no está en el JSON.")
        param_values = config_data[varied_param]
    else:
        varied_param = None
        for k, v in config_data.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float, list)):
                varied_param = k
                param_values = v
                break
        if varied_param is None:
            raise ValueError("No se encontró parámetro variable.")

    # Crear lista de configuraciones
    config_list = []

    # por cada valor en la lista con variaciones, crea copia del dicc,
    for val in param_values:
        cfg = copy.deepcopy(config_data)
        # indexa la copia del diccionario con el parametro variado para obtener
        # la lista con n variaciones y reemplaza la lista por
        # el valor var que itera el ciclo.
        cfg[varied_param] = val
        # ese nuevo diccionario lo agrega a la lista de simulaciones a hacer
        config_list.append(cfg)

    all_signals = []
    mic_positions_list = []
    fs_list = []
    valid_param_values = []
    ground_truth_angles = []

    # se itera sobre los n diccionarios en config list
    for cfg in config_list:
        try:
            room_dim = cfg["room_dim"]
            rt60 = float(cfg["rt60"]) if isinstance(cfg["rt60"], list) else float(cfg["rt60"])
            mic_amount = int(cfg["mic_amount"])  # 🔴 fuerza entero
            mic_start = cfg["mic_start"]
            mic_dist = cfg["mic_dist"]
            source_pos = cfg["source_pos"]
            fs = int(cfg["fs"]) if isinstance(cfg["fs"], list) else int(cfg["fs"])
            snr = cfg["snr"]

            # Validar cantidad mínima de micrófonos
            if mic_amount < 2:
                print(f"Skipping config with insufficient microphones: mic_amount={mic_amount}")
                continue

            # Validar source_pos (si el parámetro variable es una coordenada 3D)
            if not (isinstance(source_pos, list) and len(source_pos) == 3):
                print(f"Skipping invalid source_pos: {source_pos}")
                continue

            # Validar que la fuente esté dentro del recinto
            if not all(0 <= source_pos[i] <= room_dim[i] for i in range(3)):
                print(f"Source position out of bounds: {source_pos}")
                continue

            mic_pos = sim.mic_array(mic_amount, mic_start, mic_dist)
            room = sim.room_sim(room_dim, rt60, mic_pos, source_pos, signal_data, fs)

            # Validar que los micrófonos estén dentro del recinto
            if np.any(mic_pos[0] < 0) or np.any(mic_pos[0] > room_dim[0]) or \
               np.any(mic_pos[1] < 0) or np.any(mic_pos[1] > room_dim[1]) or \
               np.any(mic_pos[2] < 0) or np.any(mic_pos[2] > room_dim[2]):
                print(f"Mic positions out of room bounds. Skipping configuration.")
                continue
            # Construir el recinto
            

            if room.mic_array.signals.shape[1] == 0:
                print(f"Empty signals for source_pos={source_pos}, skipping.")
                continue

            signals = room.mic_array.signals

            # Agrega ruido con el SNR especificado
            signals = gen.add_awgn(signals, snr)

            # Guardar resultados válidos
            all_signals.append(signals)
            mic_positions_list.append(mic_pos)
            fs_list.append(fs)
            valid_param_values.append(cfg[varied_param])
            ground_truth_angles.append(true_doa(mic_pos, source_pos))

        except Exception as e:
            print(f"Error with config {cfg}: {e}")
            continue

    # Calcular TDOAs y DOAs con cálculo automático de max_tau si no se pasa
    all_tdoas = batch_gcc_tdoas(
        all_signals, fs_list,
        mic_positions_list=mic_positions_list,
        mic_pairs=mic_pairs,
        method=method,
        max_tau=max_tau,
        c=c
    )
    doa_results = batch_doas(all_tdoas, mic_positions_list, mic_pairs, c)

    if return_error:
        doa_errors = []
        for est, real in zip(doa_results, ground_truth_angles):
            # si el resultado es una lista de valores (por ejemplo varios DOAs por sim)
            if isinstance(est, (list, np.ndarray)) and len(est) > 0:
                error = abs(est[0] - real) % 360
                if error > 180: error = 360 - error
                doa_errors.append(error)
            elif isinstance(est, (int, float)):
                error = abs(est - real) % 360
                if error > 180: error = 360 - error
                doa_errors.append(error)
            else:
                doa_errors.append(np.nan)
        return valid_param_values, doa_errors
    else:
        return valid_param_values, doa_results




def batch_mean_std(x_data, y_data, batch_size):
    """
    Groups x_data and y_data into batches of size batch_size, then calculates the mean and 
    standard deviation of y_data, as well as the mean of x_data to position the bars.

    Parameters
    ----------
    x_data : list or np.ndarray
        X-axis values corresponding to y_data.
    y_data : list or np.ndarray
        Y-axis values.
    batch_size : int
        Size of each batch.

    Returns
    -------
    tuple of (list, list, list)
        - mean_x: mean of x_data for each batch (bar position).
        - mean_y: mean of y_data for each batch.
        - std_y: standard deviation of y_data for each batch.
    """
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    n = len(y_data)
    mean_x = []
    mean_y = []
    std_y = []

    for start_idx in range(0, n, batch_size):
        x_batch = x_data[start_idx:start_idx + batch_size]
        y_batch = y_data[start_idx:start_idx + batch_size]

        mean_x.append(np.mean(x_batch))
        mean_y.append(np.mean(y_batch))
        std_y.append(np.std(y_batch))

    return mean_x, mean_y, std_y


"""

dicc_base = {
    "room_dim": [10, 10, 10], 
    "rt60": 0.5,
    "mic_amount": 4,
    "mic_start": [1, 1, 1],
    "mic_dist": 0.1,
    "source_pos": [5, 5, 1],
    "fs": 44100}



sim.expand_param(dicc_base, "rt60", 0.05, filename = "x")
sim.expand_param(dicc_base, "source_pos", [0.05,0,0], filename = "z", n=100)
"""

"""
dicc_base = {
    "room_dim": [5, 5, 5], 
    "rt60": 0.5,
    "mic_amount": 4,
    "mic_start": [0, 0, 2.5],
    "mic_dist": 0.1,
    "source_pos": [2.5, 2.5, 2.5],
    "fs": 48000}


sim.expand_param(dicc_base, "rt60", 0.05, filename = "x")

x, audio = gen.unit_impulse((0, 88200), 44100)

full_doa_pipeline(dicc_base, audio)
"""
