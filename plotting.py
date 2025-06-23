#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 12:13:32 2023

@author: franp
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import soundfile as sf
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def plot_signals(*signals, xlabel="", ylabel="", xlim=None, ylim=None, xlogscale=False, ylogscale=False, stem=False, bars=False, categories=None, x_rot=0, y_rot=0, plot_path=None, size=(10,6), lab_fontsize=10, tit_fontsize=12, grid=False, ybase=1.0):
    """
    Plot one or more signals that share equal magnitudes in the same figure, 
    allowing for specific placement of signals in desired subplot.
    
    Parameters:
    -----------
    *signals : tuple
        Accepts tuples containing signals to plot, and where to do so. 
        Each signal should be a tuple consisting of seven elements: 
        two arrays representing the x-axis and y-axis data respectively, 
        an integer starting from 1, which indicates the graph number where the 
        signal is going to be plotted on, the plot title, the plot color, a label, and lastly, 
        a list which corresponds of y-axis errors, with length the same as the y-axis array.
        If only one plot is desired, number 1 must still be input in tuple. If no
        plot title is desired, "" must be entered. If no specific color is 
        desired, None must be entered. If no label is desired, None must be input. 
        If no y errors are desired, None must be entered.
    xlabel : str
        The label of the x-axis of the last plot.
    ylabel : str
        The label of the y-axis of the whole figure.
    xlim : tuple, optional
        A range of values for the x-axis (default None).
    ylim : tuple, optional
        A range of values for the y-axis (default None).
    xlogscale : bool, optional
        If set to True, x-axis scale will be a logarithmic one. Default is False.
    ylogscale : bool, optional
        If set to True, y-axis scale will be a logarithmic one. Default is False.
    stem : bool, optional
        If set to True, plot will be a stem plot instead of a continuous one. Default is False.
    bars : bool, optional
        If set to True, plot will be a bar plot instead of a continuous one. Default is False.
    categories : list, optional
        If bars is set to True, categories is a list of xticklabels. Default is None
    x_rot : int, optional
        Specifies xticklabel rotation degree. Default is 0.
    y_rot : int, optional
        Specifies yticklabel rotation degree. Default is 0.
    plot_path : str, optional
        Contains directory path to save a .png image of plot. 
        If left blank, plot won't be saved. If only filename is given, .png file 
        will be saved in current directory. Default is False.
    size : tuple, optional
        Tuple containing figure size in inches. Default is (10, 6).
    lab_fontsize : int, optional
        Sets x and y label font sizes. Default is 10.
    tit_fontsize : int, optional
        Sets subplot title font sizes. Default is 12.
    grid : bool, optional
        Dictates whether grid is on or off on all subplots. Default is False.
    ybase : int, optional
        Dictates ytick increments. Default is 1.0.
        
    Returns:
    -------
    None.  
    """  
    
    loc = plticker.MultipleLocator(base=ybase)
    last_elements = []
    for i, signal in enumerate(signals):
        last_elements.append([signal[2]])
        max_last_element = np.max(last_elements)
    #if only one subplot is desired 
    if max_last_element == 1:
        fig, axs = plt.subplots(figsize=size)
        for i, signal in enumerate(signals):
            if stem == False:
                if bars == True:
                    axs.bar(signal[0], signal[1], color=signal[4], label=signal[5], yerr=signal[6], ecolor = "r" if i==1 else "black", alpha = 0.8 if i==0 else 0.3, edgecolor = "black")
                else:
                    axs.plot(signal[0], signal[1], color=signal[4], label=signal[5])
            else:
                axs.stem(signal[0], signal[1], linefmt=signal[4], label=signal[5])
            if xlim:
                axs.set_xlim(xlim)
            if ylim:
                axs.set_ylim(ylim)
            axs.set_title(signal[3], fontsize=tit_fontsize)
            if not categories == None:
                axs.set_xticks(signal[0])
                axs.set_xticklabels(categories)
            axs.tick_params(axis='x', rotation=x_rot)
            axs.tick_params(axis='y', rotation=y_rot)
            axs.set_xlabel(xlabel, fontsize=lab_fontsize) 
            axs.set_ylabel(ylabel, fontsize=lab_fontsize)
             # this locator puts ticks at regular intervals
            axs.yaxis.set_major_locator(loc)
            if not signal[5] == "":
                axs.legend()
            if bars:
                axs.grid(grid, axis="y")
            else:
                axs.grid(grid)
            if xlogscale==True:
                axs.set_xscale("log")
            if ylogscale==True:
                axs.set_yscale("log")

    #if more subplots are desired        
    else:
        fig, axs = plt.subplots(max_last_element, 1, sharex=True, figsize=size)                
        for i, signal in enumerate(signals):
            if stem == False:
                if bars == True:
                        axs[signal[2]-1].bar(signal[0], signal[1], color=signal[4], label=signal[5], yerr=signal[6], edgecolor = "black")
                        axs[signal[2]-1].set_title(signal[3], fontsize=tit_fontsize)
                        axs[signal[2]-1].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
                        if not categories == None:
                            axs[signal[2]-1].set_xticks(signal[0])
                            axs[signal[2]-1].set_xticklabels(categories)
                        axs[signal[2]-1].tick_params(axis='x', rotation=x_rot)
                        axs[signal[2]-1].tick_params(axis='y', rotation=y_rot)
                        if not signal[5] == "":
                            axs[signal[2]-1].legend()
                        axs[signal[2]-1].grid(grid)
                else:
                    #for i, signal in enumerate(signals):
                        axs[signal[2]-1].plot(signal[0], signal[1], color=signal[4], label=signal[5])

                        axs[signal[2]-1].set_title(signal[3], fontsize=tit_fontsize)
                        axs[signal[2]-1].tick_params(axis='x', rotation=x_rot, which='both', bottom=True, top=False, labelbottom=True)
                        axs[signal[2]-1].tick_params(axis='y', rotation=y_rot)
                        if not signal[5] == "":
                            axs[signal[2]-1].legend()
                        axs[signal[2]-1].grid(grid)
                    
            else:
                #for i, signal in enumerate(signals):
                    axs[signal[2]-1].stem(signal[0], signal[1], linefmt=signal[4], label=signal[5])
                    axs[signal[2]-1].set_title(signal[3], fontsize=tit_fontsize)
                    axs[signal[2]-1].tick_params(axis='x', rotation=x_rot, which='both', bottom=True, top=False, labelbottom=True)
                    axs[signal[2]-1].tick_params(axis='y', rotation=y_rot)
                    if not signal[5] == "":
                        axs[signal[2]-1].legend()
                    axs[signal[2]-1].grid(grid)
            if xlogscale==True:
                axs[signal[2]-1].set_xscale("log")
            if ylogscale==True:
                axs[signal[2]-1].set_yscale("log")
            axs[signal[2]-1].yaxis.set_major_locator(loc)
        for j in range(int(max_last_element)):
            if xlim:
                axs[j-1].set_xlim(xlim)
            if ylim:
                axs[j-1].set_ylim(ylim)
        axs[-1].set_xlabel(xlabel, fontsize=lab_fontsize) 

        fig.text(0.000001, 0.5, ylabel, va="center", rotation="vertical", fontsize=lab_fontsize)
    fig.tight_layout()
    if not plot_path is None:
        plt.savefig(plot_path)
    plt.show()        
    return

def spectrum(signal, sr=16000, title="Frequency Spectrum", xlim=(20, 22050), ylim=None, size=(10, 6), color=None, ylabel="Level [dB]"):

    """
    Display mono frequency spectrum of a given signal.

    Parameters:
    ----------
    signal : str, np.ndarray
        File path of the input signal or array containing amplitude values.
    sr : int, optional
        Desired samplerate in Hz of input signal. If path is given, signal samplerate
        will be used instead. Default is 16000.
    title : str, optional
        Plot title. Default is "Frequency Spectrum".
    xlim : tuple, optional
        Defines limits of x-axis in Hz. Default is (20, 22050).
    ylim : tuple, optional
        Defines limits of y-axis in dB. Default is None.
    size : tuple, optional
        Tuple containing figure size in inches. Default is (10, 6).
    color : str, optional
        Plot color. Default is None.
    ylabel : str, optional
        Y-axis label. Default is "Level [dB]".
        
    Returns:
    -------
    None.
    """
    
    if type(signal) == str:
        audio, sr = sf.read(signal)
    else:
        audio = signal
        if sr==0:
            raise ValueError("Desired sampling rate must be input")
    n_channels = np.shape(audio)
    if not len(n_channels) == 1: 
        audio = np.sum(audio, axis=1)
    fft_raw = np.fft.fft(audio)
    fft = fft_raw[:len(fft_raw)//2]
    # lose the sample scaling info and divide by half the sample amount
    fft_mag = abs(fft) / len(fft)
    
    freq_ax = np.linspace(0, sr/2, len(fft), endpoint=False)
    fft_mag_norm = fft_mag / np.max(abs(fft_mag))
    eps = np.finfo(float).eps
    fft_mag_db = 20*np.log10(fft_mag_norm + eps)
    
    # create plot
    fig, ax = plt.subplots(1, 1, figsize=size)
    ax.set_xscale("log")
    ax.grid()
    ax.plot(freq_ax, fft_mag_db, color=color)
    ax.set_title(title, fontsize=17)
    ax.set_xlabel("Frequency [Hz]", fontsize=12) 
    ax.set_ylabel(ylabel, fontsize=12) 
    # change xtick labels
    new_xticks = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
    new_xticklabels = ["63", "125", "250", "500", "1000", "2000", "4000", "8000"]
    ax.set_xticks(new_xticks)
    ax.set_xticklabels(new_xticklabels)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()
    return

def plot_room_top_view(room_dim, mic_positions, source_position, zoom=True, zoom_margin=0.5):
    """
    Plots a top-down 2D view of the room, microphone positions, and the sound source,
    with optional automatic zoom on the microphone array.

    Parameters
    ----------
    room_dim : list or array-like
        Dimensions of the room in meters [length_x, width_y, height_z].
    mic_positions : np.ndarray
        Microphone positions, shape (3, N) or (N, 3). Each row is [x, y, z].
    source_position : list or array-like
        Source position [x, y, z].
    zoom : bool, optional
        Whether to include a zoomed inset around the mic array. Default is True.
    zoom_margin : float, optional
        Margin (in meters) around the mic array in the zoom window. Default is 0.5.
    """
    # Ensure shape is (N, 3)
    mic_positions = np.atleast_2d(mic_positions)
    if mic_positions.shape[0] == 3 and mic_positions.shape[1] != 3:
        mic_positions = mic_positions.T

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, room_dim[0])
    ax.set_ylim(0, room_dim[1])
    ax.set_aspect('equal')
    ax.grid(True)

    # Dibujar paredes
    ax.plot([0, room_dim[0], room_dim[0], 0, 0],
            [0, 0, room_dim[1], room_dim[1], 0],
            'k-', label='Paredes de la sala')

    # Dibujar micrófonos
    for i, (x, y, _) in enumerate(mic_positions):
        ax.plot(x, y, 'bo', markersize=4)

    # Dibujar fuente
    ax.plot(source_position[0], source_position[1], 'r*', markersize=12, label='Fuente sonora')
    ax.text(source_position[0] + 0.2, source_position[1] + 0.2, 'Fuente', fontsize=9)

    # Inset automático alrededor de micrófonos
    if zoom:
        x_coords = mic_positions[:, 0]
        y_coords = mic_positions[:, 1]
        x0, x1 = x_coords.min() - zoom_margin, x_coords.max() + zoom_margin
        y0, y1 = y_coords.min() - zoom_margin, y_coords.max() + zoom_margin

        axins = inset_axes(ax, width="40%", height="40%", loc='lower right')
        axins.set_xlim(x0, x1)
        axins.set_ylim(y0, y1)

        for i, (x, y, _) in enumerate(mic_positions):
            axins.plot(x, y, 'bo', markersize=8)
            axins.text(x, y + 0.05, f'{i}', fontsize=6)

        axins.set_xticks([])
        axins.set_yticks([])
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax.set_title('Vista superior de la sala')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    plt.show()