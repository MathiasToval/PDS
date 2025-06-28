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
    from the center of the microphone array to the source position.

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
    mic_center = np.mean(mic_pos, axis=1)  # geometric center of array
    src_vec = np.array(source_pos[:2]) - mic_center[:2]  # vector from center to source in XY

    angle_rad = np.arctan2(src_vec[1], src_vec[0])  # atan2(y, x)
    angle_deg = np.degrees(angle_rad) % 360

    return angle_deg


def gcc_tdoas(signals, fs, mic_pairs=None, method='classicfft', max_tau=None):
    """
    Estimate the time delays of arrival (TDOAs) between multiple pairs of signals 
    using Generalized Cross Correlation (GCC) with various weighting methods.

    Parameters
    ----------
    signals : ndarray
        2D array with shape (n_mics, n_samples), containing one signal per microphone.
    fs : int or float
        Sampling frequency in Hz.
    mic_pairs : list of tuple of int, optional
        List of microphone index pairs [(i, j), ...] for which to compute the TDOAs.
        If None, all adjacent mic pairs are used: [(0,1), (1,2), ...].
    method : str, optional
        GCC weighting method. Options include 'phat', 'scot', 'roth', 'eckart', 'ml', and 'classic'.
    max_tau : float, optional
        Maximum expected time delay in seconds. Limits the search range.

    Returns
    -------
    tdoas : list of float
        Estimated TDOAs (in seconds) for each mic pair.

    Raises
    ------
    ValueError
        If an invalid method string is provided or signals shape is invalid.
    """
    n_mics, n_samples = signals.shape

    if mic_pairs is None:
        mic_pairs = [(i, i+1) for i in range(n_mics - 1)]

    tdoas = []

    for i, j in mic_pairs:
        sig1 = signals[i]
        sig2 = signals[j]
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
            lags = np.arange(len(cc)) - (len(cc) // 2)

        t_lags = lags / fs

        if max_tau is not None:
            mask = np.abs(t_lags) <= max_tau
            cc = cc[mask]
            t_lags = t_lags[mask]

        tdoa = t_lags[np.argmax(cc)]
        tdoas.append(tdoa)

    return tdoas


def doa(tdoa, mic_positions, mic_pairs=None, c=343, return_all=False):
    """
    Calculates Direction of Arrival (DOA) angles from Time Differences of Arrival (TDOA) 
    using multiple microphone pairs.

    Parameters
    ----------
    tdoa : array-like
        TDOAs in seconds, one per mic pair.
    mic_positions : np.ndarray
        Array of shape (3, n_mics), each column is [x, y, z] of one mic.
    mic_pairs : list of tuple of int, optional
        List of (i, j) mic index pairs corresponding to each specified tdoa.
        If None, uses adjacent pairs.
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
    """
    tdoa = np.asarray(tdoa)
    mic_positions = np.asarray(mic_positions)

    if mic_positions.shape[0] != 3:
        raise ValueError("mic_positions must have shape (3, n_mics) — 3 rows for x, y, z.")

    n_mics = mic_positions.shape[1]

# asume que los tdoa son de pares adyacentes
    if mic_pairs is None:
        mic_pairs = [(i, i + 1) for i in range(n_mics - 1)]

    if len(tdoa) != len(mic_pairs):
        raise ValueError("Number of TDOAs must match number of mic pairs.")

    doa_angles = []
    #zip recorre (itera) los valores de cada array
    for tau, (i, j) in zip(tdoa, mic_pairs):
        d_vec = mic_positions[:, j] - mic_positions[:, i] #resta la posicion del mic j a la del i
        d = np.linalg.norm(d_vec) #ese vector resta al sacarle el modulo es la distancia entre mics
        cos_theta = np.clip(tau * c / d, -1, 1)
        angle = np.degrees(np.arccos(cos_theta))
        doa_angles.append(angle)

    doa_angles = np.array(doa_angles)

    return doa_angles if return_all else round(doa_angles.mean(), 2)


def batch_gcc_tdoas(all_signals, fs_list, mic_positions_list=None, mic_pairs=None, method='classicfft', max_tau=None, c=343):
    """
    Computes TDOAs for a list of simulations.

    Parameters
    ----------
    all_signals : list of np.ndarray
        List of (n_mics, n_samples) arrays, one per simulation.
    fs_list : list of int or float
        Sampling frequencies, one per simulation.
    mic_positions_list : list of np.ndarray, optional
        List of mic position arrays, needed to compute max_tau automatically if not given.
    mic_pairs : list of tuple of int, optional
        Microphone index pairs. If None, uses adjacent pairs.
    method : str, optional
        GCC method. Default is 'classic'.
    max_tau : float, optional
        Maximum TDOA to consider (in seconds). If None, will be calculated from mic_positions_list.
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
            if mic_pairs is None:
                n_mics = mic_pos.shape[1]
                mic_pairs_local = [(i, i + 1) for i in range(n_mics - 1)]
            else:
                mic_pairs_local = mic_pairs
            max_dist = max(np.linalg.norm(mic_pos[:, i] - mic_pos[:, j]) for i, j in mic_pairs_local)
            max_tau_sim = max_dist / c
        else:
            max_tau_sim = max_tau

        tdoas = gcc_tdoas(signals, fs, mic_pairs=mic_pairs, method=method, max_tau=max_tau_sim)
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
        Mic index pairs.
    c : float, optional
        Speed of sound.

    Returns
    -------
    list of float or list of np.ndarray
        DOA (degrees) per simulation, either average or all angles.
    """
    doas_results = []
    for tdoas, mic_pos in zip(all_tdoas, mic_positions_list):
        doa_value = doa(tdoas, mic_pos, mic_pairs=mic_pairs, c=c, return_all=False)
        doas_results.append(doa_value)
    return doas_results



def full_doa_pipeline(json_path, signal, mic_pairs=None, method='classicfft', max_tau=None, c=343, variable_param=None, return_error=True, mic_noise_snr=None):
    """
    Carga configuraciones, simula, calcula TDOAs y DOAs, con opción de agregar ruido a micrófonos.

    Parameters
    ----------
    json_path : str
        Ruta al archivo JSON con parámetros de simulación.
    signal : str o np.ndarray
        Señal fuente o ruta a archivo .wav.
    mic_pairs : list of tuple of int, optional
        Pares de micrófonos para TDOA/DOA.
    method : str, optional
        Método de GCC ('classic', 'phat', etc.).
    max_tau : float, optional
        Máximo TDOA esperado.
    c : float, optional
        Velocidad del sonido.
    variable_param : str, optional
        Nombre del parámetro que varía.
    return_error : bool, optional
        Si True, devuelve error absoluto con DOA real.
    mic_noise_snr : float, optional
        Si se especifica, agrega ruido blanco gaussiano a los micrófonos con esa SNR en dB.

    Returns
    -------
    tuple of (list, list, str)
        - Valores del parámetro que varió (x-axis).
        - DOAs estimados o errores.
        - Nombre del parámetro variado.
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
    # agarra e indexa la lista con n variaciones
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

    #por cada valor en la lista con variaciones, crea copia del dicc,
    for val in param_values:
        cfg = copy.deepcopy(config_data)
        #indexa la copia del diccionario con el parametro variado para obtener
        #la lista con n variaciones y reemplaza la lista por
        #el valor var que itera el ciclo.
        cfg[varied_param] = val
        #ese nuevo diccionario lo agrega a la lista de simulaciones a hacer
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
            rt60 = cfg["rt60"]
            mic_amount = cfg["mic_amount"]
            mic_start = cfg["mic_start"]
            mic_dist = cfg["mic_dist"]
            source_pos = cfg["source_pos"]
            fs = cfg["fs"]

            # se itera sobre los n diccionarios en config list
            if isinstance(rt60, list): rt60 = float(rt60[0])
            if isinstance(fs, list): fs = int(fs[0])

            # Validar source_pos (si el parámetro variable es una coordenada 3D)
            if not (isinstance(source_pos, list) and len(source_pos) == 3):
                print(f"Skipping invalid source_pos: {source_pos}")
                continue

            # Construir el recinto# Construir el recinto
            mic_pos = sim.mic_array(mic_amount, mic_start, mic_dist)
            room = sim.room_sim(room_dim, rt60, mic_pos, source_pos, signal_data, fs)

            if room.mic_array.signals.shape[1] == 0:
                print(f"Empty signals for source_pos={source_pos}, skipping.")
                continue

            signals = room.mic_array.signals

            # 🎯 Si hay SNR definido, agregá ruido
            if mic_noise_snr is not None:
                signals = gen.add_awgn(signals, mic_noise_snr)

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
                # corregir si el error es mayor a 180 (ángulo circular)
                if error > 180: error = 360 - error
                doa_errors.append(error)
            elif isinstance(est, (int, float)):
                error = abs(est - real) % 360
                if error > 180: error = 360 - error
                doa_errors.append(error)
                # si es un único número
            else:
                doa_errors.append(np.nan)
        return valid_param_values, doa_errors
    else:
        return valid_param_values, doa_results


"""

dicc_base = {
    "room_dim": [10, 10, 10], 
    "rt60": 0.5,
    "mic_amount": 4,
    "mic_start": [1, 1, 1],
    "mic_dist": 0.1,
    "source_pos": [5, 5, 1],
    "fs": 44100}

x, audio = gen.unit_impulse((0, 88200), 44100)


sim.expand_param(dicc_base, "rt60", 0.05, filename = "x")
sim.expand_param(dicc_base, "source_pos", [0.05,0,0], filename = "z", n=100)
"""
