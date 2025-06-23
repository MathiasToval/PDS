#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 17:42:16 2025

@author: franp
"""

import numpy as np
from scipy.signal import correlate
from scipy.fft import fft, ifft


def estimate_tdoa(sig1, sig2, fs):
    corr = correlate(sig1, sig2, mode='full')
    lag = np.argmax(corr) - (len(sig2) - 1)
    return lag / fs  # tiempo en segundos

def corr_norm(sig1, sig2, fs):
    corr = correlate(sig1, sig2, mode='full')
    lag = (np.argmax(corr) - (len(sig2) - 1))/fs
    return lag, corr/np.abs(np.max(corr))

def gcc_tdoas(signals, fs, mic_pairs=None, method='classic', max_tau=None):
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

    # Si no se especifican pares, se usan pares consecutivos
    if mic_pairs is None:
        mic_pairs = [(i, i+1) for i in range(n_mics - 1)]

    tdoas = []

    for i, j in mic_pairs:
        sig1 = signals[i]
        sig2 = signals[j]

        n = len(sig1) + len(sig2) - 1  # ajustar longitud total

        if method == 'classic':
            cc = correlate(sig1, sig2, mode='full')
        else:
            SIG1 = fft(sig1, n=n)
            SIG2 = fft(sig2, n=n)  # zero padding para lograr que la convolución circular sea igual a la lineal
            G = SIG1 * np.conj(SIG2)  # Espectro de la correlación cruzada

            P1 = np.abs(SIG1) ** 2
            P2 = np.abs(SIG2) ** 2

            if method == 'phat':
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
                raise ValueError("Invalid method. Choose from 'phat', 'scot', 'roth', 'eckart', 'ml', or 'classic'.")

            G_weighted = G * W
            cc = np.real(ifft(G_weighted))
            cc = np.fft.fftshift(cc)
            # centra la corr alrededor del lag 0, ya que el orden de ese resultado 
            # sigue siendo circular, en el sentido de cómo se almacenan los lags en el array.

        # eje de lags como en correlate, se divide por dos ya que n es la long de la
        # convolución y la mitad entera lo que se llama en la correlación común como
        # len(sig2)-1
        lags = np.arange(-(n // 2), (n + 1) // 2)
        t_lags = lags / fs

        if max_tau is not None:
            # puede servir para ignorar los resultados por fuera del máximo retardo esperado
            # el cual ocurre cuando cos = 1
            mask = np.abs(t_lags) <= max_tau
            cc = cc[mask]
            t_lags = t_lags[mask]

        # este es el valor con convención correcta
        tdoa = t_lags[np.argmax(cc)]
        tdoas.append(tdoa)

    return tdoas


def gcc(sig1, sig2, fs, method='classic', norm=False):
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
        GCC weighting method. Options: 'phat', 'scot', 'roth', 'eckart', 'ml', 'classic'.
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

    if method == 'classic':
        cc = correlate(sig1, sig2, mode='full')
    else:
        SIG1 = fft(sig1, n=n)
        SIG2 = fft(sig2, n=n)
        G = SIG1 * np.conj(SIG2)

        P1 = np.abs(SIG1) ** 2
        P2 = np.abs(SIG2) ** 2

        if method == 'phat':
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
            raise ValueError("Invalid method. Choose from 'phat', 'scot', 'roth', 'eckart', 'ml', or 'classic'.")

        G_weighted = G * W
        cc = np.real(ifft(G_weighted))
        cc = np.fft.fftshift(cc)

    lags = np.arange(-(n // 2), (n + 1) // 2)
    t_lags = lags / fs
    
    if norm:
        cc = cc/np.max(np.abs(cc))

    return t_lags, cc


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
