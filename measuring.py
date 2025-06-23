import numpy as np
import modifysignals as mod
import measuring as meas
from scipy import stats

def SPL(pa, pa_ref=0.00002):
    
    """
    Compute the sound pressure level in dB for a given reference value.
    
    Parameters
    ----------
    pa : float, np.ndarray
        Pressure [Pa]. If an array is input, it's elements must be float type data.
    pa_ref : float, optional
        Reference value. Default is 20 uPa.
    
    Returns
    -------
    SPL : float, np.ndarray
        Sound pressure level.
    
    """
    SPL = 10*np.log10(((pa + np.finfo(float).eps)/pa_ref)**2)
    
    return SPL

def SPL_ave(SPL):
    
    """
    Calculates the sound pressure level average by definition.
    
    Parameters
    ----------
    
    SPL : NUMPY ARRAY
        Array of sound pressure levels to evaluate.
    
    Returns
    -------
    
    SPL_ave : FLOAT
        Average sound pressure level.
    
    """
    
    N = len(SPL)
    SPL_sum = 0
    
    for SPL_i in SPL:
        SPL_sum = SPL_sum + 10**(SPL_i/20)
    
    SPL_ave = 20*np.log10(SPL_sum/N)
    
    return SPL_ave

def band_average(SPL_matrix):
    """
    Calculates the average sound pressure level per band for multiple takes of the same 
    audio source.
    
    Parameters
    ----------
    SPL_matrix : np.ndarray
        Matrix of equivalent SPL values per frequency band.
        Each row of the matrix represents a frequency level, and each column represents an audio source.

    Returns
    -------
    band_ave : list
        List containing the equivalent SPL per frequency band, averaged over multiple takes.
        
    """
    num_bands = SPL_matrix.shape[1]
    band_ave = []

    for j in range(num_bands):
        band = SPL_matrix[:, j]
        band_mean = SPL_ave(band)
        band_ave.append(band_mean)
    
    return band_ave

def LZeq(pa):
    """
    Compute the equivalent sound pressure level in dBSPL with Z weighting curve.
    Time integration will be determined by amount of sound pressure samples
    
    Parameters
    ----------
    pa : np.ndarray
        Pressure [Pa]. Array elements must be float type data.
    pa_ref : float, optional
        Reference value. Default is 20 uPa.
    
    Returns
    -------
    Leq : float
        Sound pressure level.
    
    """
    Leq = meas.SPL(mod.RMS(pa))
    return Leq

def std(*data):
    """
    Compute the standard deviation from between elements on the same position/index for a variable number of 1-D arrays.

    Parameters
    ----------
    data : np.ndarray
        Variable number of 1-D arrays, which need to be the same length.

    Output
    ------
    array_std : np.ndarray
        1-D array with the standard deviation of the elements of each index.
    """
    arr_len = len(data[0])
    array_std = np.zeros(len(data[0]))
    
    for h in range(arr_len):
        elem_list = []

        for i, value in enumerate(data):
            elem_list.append(value[h])
   
        std_values = np.std(elem_list)
        array_std[h] = std_values

    return array_std