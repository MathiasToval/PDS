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

def gcc_tdoa(sig1, sig2, fs, method='phat', max_tau=None):
    
    """
    Estima el retardo entre sig1 y sig2 con signo consistente con scipy.signal.correlate.
    
    Estima el retardo entre dos señales usando GCC.

    Parámetros:
        sig1, sig2 : señales de entrada (arrays 1D).
        fs : frecuencia de muestreo [Hz].
        method : 'phat', 'scot', 'roth', o 'classic'.
        max_tau : retardo máximo esperado [s], opcional.

    Retorna:
        Retardo estimado en segundos (float).
    """
    n = len(sig1) + len(sig2) - 1  # ajustar longitud total

    if method == 'classic':
        cc = correlate(sig1, sig2, mode='full')
    else:
        SIG1 = fft(sig1, n=n)
        SIG2 = fft(sig2, n=n) #zero padding para lograr que la convolucion circular sea igual a la lineal
        G = SIG1 * np.conj(SIG2) #Espectro de la correlación cruzada

        P1 = np.abs(SIG1) ** 2
        P2 = np.abs(SIG2) ** 2

        if method == 'phat':
            W = 1 / (np.abs(G) + 1e-12)
        elif method == 'scot':
            W = 1 / (np.sqrt(P1 * P2) + 1e-12)
        elif method == 'roth':
            W = 1 / (P2 + 1e-12)
        else:
            raise ValueError("Método inválido. Usar 'phat', 'scot', 'roth' o 'classic'.")

        G_weighted = G * W
        cc = np.real(ifft(G_weighted))
        cc = np.fft.fftshift(cc) 
        #centra la corr alrededor del lag 0, ya que el orden de ese resultado 
        #sigue siendo circular, en el sentido de cómo se almacenan los lags en el array.
    # eje de lags como en correlate, se divide por dos ya que n es la long de la
    # convolución y la mitad entera lo que se llama en la correlación comun como
    # len(sig2)-1
    lags = np.arange(-(n // 2), (n + 1) // 2)
    t_lags = lags / fs

    if max_tau is not None:
        # puede servir para ignorar los resultados por fuera del máximo retardo esperado
        mask = np.abs(t_lags) <= max_tau
        cc = cc[mask]
        t_lags = t_lags[mask]

    # este es el valor con convención correcta
    return t_lags[np.argmax(cc)]


def doa(tdoa, d, c=343):
    
    """
   Calculates the Direction Of Arrival (DOA) angle in degrees from the Time Difference Of Arrival (TDOA)
   between two sensors.

   Parameters:
       tdoa : float or array-like
           Time difference of arrival between two signals (in seconds).
       d : float
           Distance between the two sensors (e.g., microphones) in meters.
       c : float
           Propagation speed of the signal in the medium (e.g., speed of sound in m/s).
           Defaults to 343 m/s.

   Returns:
       float or np.ndarray
           Estimated arrival angle in degrees, where 0° corresponds to the direction aligned
           with the line between the sensors, and ±90° is perpendicular to it.

   Notes:
       - Uses the geometric relation: cos(θ) = (τ * c) / d.
       - Applies clipping (`np.clip`) to constrain values within [-1, 1] to avoid numerical errors.
   """
   
    cos_theta = tdoa * c / d
    cos_theta = np.clip(cos_theta, -1, 1)
    return np.degrees(np.arccos(cos_theta))

