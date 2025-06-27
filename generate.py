import numpy as np
import math


def unit_impulse(domain, centre):
    """
    Generate a unit impulse signal.
    
    Parameters:
    -----------
    domain : tuple
        Tuple containing signal domain range, must have int type limits. First 
        limit must be less or equal than the second one.
    centre : int
        Defines where impulse is.

    Returns:
    --------
    arr_x : np.ndarray
        An array containing the sample values.
    arr_y : np.ndarray
        An array which contains the amplitude values of generated signal.
        
    Raises:
    -------
    ValueError
        If domain is not a tuple or if domain range limits are not integers.
    """

    if not type(domain) == tuple:
        raise ValueError("domain must be a tuple")
    if not type(domain[0]) == int or not type(domain[1]) == int:
        raise ValueError("domain range limits must be integers") 
        
    arr_x = np.arange(domain[0], domain[1]+1)
    arr_y = np.zeros(len(arr_x))
    index = np.where(arr_x == centre)[0]
    arr_y[index] = 1
    return arr_x, arr_y


def exp_signal(domain, base=math.exp(1), n_scale=1):
    """
    Generate a discrete exponential signal.
    
    Parameters:
    -----------
    domain : tuple
        Tuple containing signal domain range, must have int type limits. First 
        limit must be less or equal than the second one.
    base : float, optional
        Base of the exponential. Default is e.
    n_scale : float, optional
        Coefficient from exponential argument. Default is 1.

    Returns:
    --------
    arr_x : np.ndarray
        Array containing signal sample values.
    arr_y : np.ndarray
        Array containing signal amplitude values.
        
    Raises:
    -------
    ValueError
        If domain is not a tuple or if domain range limits are not integers.
    """
    
    if not type(domain) == tuple:
        raise ValueError("domain must be a tuple")
    if not type(domain[0]) == int or not type(domain[1]) == int:
        raise ValueError("domain range limits must be integers") 
        
    arr_x = np.arange(domain[0], domain[1]+1)
    arr_y = (base**(n_scale*arr_x))
    return arr_x, arr_y


def unit_step(domain, centre):
    """
    Generate a discrete unit step signal.
    
    Parameters:
    -----------
    domain : tuple
        Tuple containing signal domain range, must have int type limits. First 
        limit must be less or equal than the second one.
    centre : int
        Centre of unit step to generate.

    Returns:
    --------
    arr_x : np.ndarray
        Array containing signal sample values.
    arr_y : np.ndarray
        Array containing signal amplitude values.

    Raises:
    -------
    ValueError
        If the centre value is not an integer, or the domain is not a tuple,
        or the domain range limits are not integers.
    """
    
    zeros = []
    ones = []
    
    if type(centre) == float:
        raise ValueError("centre value must be an integer")
    if not type(domain) == tuple:
        raise ValueError("domain must be a tuple")
    if not type(domain[0]) == int or not type(domain[1]) == int:
        raise ValueError("domain range limits must be integers")     

    for j in range(int(domain[0]), int(centre)):
        zeros.append(j)
    for h in range(int(centre), int(domain[1]+1)):
        ones.append(h)
    
    arr_x = np.arange(domain[0], domain[1]+1)
    arr_y = np.concatenate((np.zeros(len(zeros)), np.ones(len(ones))))
    return arr_x, arr_y


def triangular_pulse(factor, domain):
    """
    Generate a discrete triangular pulse signal centered around sample 0.
    
    Parameters:
    -----------
    factor : int
        Number of samples that represent half of the pulse base.
    domain : tuple
        Tuple containing signal domain range, must have int type limits. First 
        limit must be less or equal than the second one.
    
    Returns:
    --------
    arr_x : np.ndarray
        Array containing signal sample values.
    arr_y : np.ndarray
        Array containing signal amplitude values.
    
    Raises:
    -------
    ValueError
        If the domain is not of tuple type, or the domain range limits are not of int type data.
    """
    
    base_samp = []
    arr_x = []
    
    if not type(domain) == tuple:
        raise ValueError("domain must be a tuple")
    if not type(domain[0]) == int or not type(domain[1]) == int:
        raise ValueError("domain range limits must be integers")  
         
    base_range = ((-factor), factor)
    for i in range(int(base_range[0]), int(base_range[1]+1)):
        base_samp.append(i) 
    for i in range(int(domain[0]), int(domain[1]+1)):
        arr_x.append(i) 

    arr_y = np.concatenate((np.zeros(base_samp[0] - arr_x[0]), np.linspace(0, 1, factor, False), np.linspace(1, 0, factor+1), np.zeros(arr_x[-1] - base_samp[-1])))
    return arr_x, arr_y


def unit_pulse(domain, base_range):
    """
    Generate a discrete unit pulse train signal.
    
    Parameters:
    ----------
    domain : tuple
        Tuple containing signal domain range, must have int type limits. First 
        limit must be less or equal than the second one.
    base_range : tuple
        Tuple containing the range of the pulse sample values, must have int type limits. 
        The range must be included in the domain range.
               
    Returns:
    -------
    arr_x : np.ndarray
        Array containing signal sample values.
    arr_y : np.ndarray
        Array containing signal amplitude values.
        
    Raises:
    -------
    ValueError 
        If domain is not a tuple or if domain range limits are not integers.
    ValueError 
        If base_range is not a tuple or if base_range limits are not integers.
    """
    
    if not type(domain) == tuple:
        raise ValueError("domain must be a tuple")
    if not type(domain[0]) == int or not type(domain[1]) == int:
        raise ValueError("domain range limits must be integers")
    if not type(base_range) == tuple:
        raise ValueError("base_range must be a tuple") 
    if not type(base_range[0]) == int or not type(base_range[1]) == int:
        raise ValueError("base_range limits must be integers")
    
    arr_x = np.arange(domain[0], domain[1]+1)
    samples = []
    for i in range(int(base_range[0]), int(base_range[1]+1)):
        samples.append(i)
        
    arr_y = np.concatenate((np.zeros(samples[0] - arr_x[0]), np.ones(len(samples)), np.zeros(arr_x[-1] - samples[-1])))    
    return arr_x, arr_y


def random_signal(median=0, std=1, domain=(-10,10)):
    """
    Generate discrete gaussian noise.
    
    Parameters:
    -----------
    median : float, optional
        Centre of normal distribution from amplitude values. Default is 0.
    std : float, optional
        Standard deviation of normal distribution from amplitude values. Specifies the dispersion of possible amplitude values. 
        If negative, its absolute value will be taken. Default is 1.
    domain : tuple, optional
        Tuple containing signal domain range, must have int type limits. First 
        limit must be less or equal than the second one. Default is (-10, 10).

    Returns:
    --------
    arr_x : np.ndarray
        Array containing signal sample values.
    arr_y : np.ndarray
        Array containing signal amplitude values.
        
    Raises:
    -------
    ValueError
        If median or std is not an integer, or if domain is not a tuple, 
        or if domain limits are not integers.
    """
    
    if not type(median) == int:
         raise ValueError("median must be an integer") 
    if not type(std) == int:
        raise ValueError("std must be an integer")      
    if not type(domain) == tuple:
        raise ValueError("domain must be a tuple") 
    if not type(domain[0]) == int or not type(domain[1]) == int:
        raise ValueError("domain range limits must be integers") 
    arr_x = np.arange(domain[0], domain[1]+1)
    arr_y = np.random.normal(median, abs(std), np.size(arr_x))
    return arr_x, arr_y


def ir(sr, t60, amp, amp_nf):
    """
    Generate an artificial impulse response from an LTI system.

    Parameters
    ----------
    sr : int
        Sample rate of artificial impulse response.
    t60 : float
        Decay time in seconds.
    amp : float
        IR amplitude factor in linear units.
    amp_nf : float
        Noise floor amplitude factor in linear units.

    Returns
    -------
    arr_x : np.ndarray
        Array containing time values of generated IR in seconds.
    arr_y : np.ndarray
        Array containing amplitude values of generated IR in linear units.
    """
    domain = (0, 2 * t60)
    arr_x = np.arange(domain[0], domain[1], 1/sr)
    noise = np.random.normal(0, 1, np.size(arr_x))
    # Calculate the decay rate tau (constant)
    tau = t60/np.log(10**3) 

    exp = np.exp(-(arr_x)/tau)
    arr_y = amp * exp * noise + amp_nf * noise
    return arr_x, arr_y

def add_awgn(signals, snr_db):
    """
    Agrega ruido blanco gaussiano (AWGN) a cada señal del array de micrófonos.

    Parameters
    ----------
    signals : np.ndarray
        Señales del micrófono. Shape (n_mics, n_samples)
    snr_db : float
        Relación señal a ruido deseada en decibelios (dB)

    Returns
    -------
    np.ndarray
        Señales con ruido añadido.
    """
    noisy_signals = np.zeros_like(signals)
    for i in range(signals.shape[0]):
        signal_power = np.mean(signals[i] ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), size=signals.shape[1])
        noisy_signals[i] = signals[i] + noise
    return noisy_signals