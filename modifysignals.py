#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 22:05:36 2023

@author: franp
"""
import numpy as np

def RMS(v):
    
    """
    Calculates the RMS value of a given vector.

    Parameters
    ----------
    v : np.ndarray
        Vector which values are going to be evaluated.

    Returns
    -------
    v_RMS : float
        RMS value.

    """
    
    N = len(v)
    v_sum = 0
    
    for v_i in v:
        v_sum = v_sum + v_i**2
    
    v_RMS = (v_sum/N)**(1/2)
    
    return v_RMS

def normalize(signal):
    """
    Scale the amplitude levels of a mono signal relative to 1,
    which is the value that the maximum level is allowed to reach.
    
    Parameters:
    -----------
    signal : np.ndarray
        Array containing amplitude values for signal. Default is None.
    signal_path : str, optional
        File path of the input signal. If a signal is given an array, then this
        variable won't be used, regardless if a path is input or not. 
        
    Output:
    -------
    signal_norm : np.ndarray
        Array which contains the amplitude values of the normalized signal.
        
    Raises:
    -------
    ValueError: 
        If neither signal nor signal_path are provided.
    """
    
    signal_max = np.max(np.abs(signal))
    signal_norm = signal / signal_max
    return signal_norm   