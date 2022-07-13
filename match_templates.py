"""
Algorithm based on the paper 'Automatic Chord Recognition from
Audio Using Enhanced Pitch Class Profile' by Kyogu Lee
This script computes 12 dimensional chromagram for chord detection
@author ORCHISAMA
"""

from __future__ import division
import numpy as np 
import os
import sys
import csv
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import json
from chromagram import compute_chroma


if __name__ == "__main__":
    """Correlate chord with existing binary chord templates to find best batch"""

    #window length, hop size
    NFFT = 8192
    HOP_SIZE = 1024

    RESULT_DIRECTORY = './'
    FILENAME = sys.argv[1]
    FILEBASENAME = os.path.splitext(os.path.basename(FILENAME))[0]
    
    CHORDS = ['N','G','G#','A','A#','B','C','C#','D','D#','E','F','F#','Gm','G#m','Am','A#m','Bm','Cm','C#m','Dm','D#m','Em','Fm','F#m']
    
    """read from JSON file to get chord templates"""
    with open('chord_templates.json', 'r') as fp:
        templates_json = json.load(fp)
    
    templates = []
    
    for chord in CHORDS:
    	if chord == 'N':
    		continue
    	templates.append(templates_json[chord])
    
    
    """read audio and compute chromagram"""
    (fs,s) = read(FILENAME)
    
    x = s[::4]
    x = x[:,1]
    fs = int(fs/4)
    
    #framing audio, omputing PCP
    nFrames = int(np.round(len(x)/(NFFT-HOP_SIZE)))
    #zero padding to make signal length long enough to have nFrames
    x = np.append(x, np.zeros(NFFT))
    xFrame = np.empty((NFFT, nFrames))
    start = 0   
    chroma = np.empty((12,nFrames)) 
    id_chord = np.zeros(nFrames, dtype='int32')
    timestamp = np.zeros(nFrames)
    max_cor = np.zeros(nFrames)
    
    for n in range(nFrames):
    	xFrame[:,n] = x[start:start+NFFT] 
    	start = start + NFFT - HOP_SIZE 
    	timestamp[n] = n*(NFFT-HOP_SIZE)/fs
    	chroma[:,n] = compute_chroma(xFrame[:,n],fs)
    	plt.figure(1)
    	plt.plot(chroma[:,n])
    
    	"""Correlate 12D chroma vector with each of 24 major and minor CHORDS"""
    	cor_vec = np.zeros(24)
    	for ni in range(24):
    		cor_vec[ni] = np.correlate(chroma[:,n], np.array(templates[ni])) 
    	max_cor[n] = np.max(cor_vec)
    	id_chord[n] =  np.argmax(cor_vec) + 1
    
    
    #if max_cor[n] < threshold, then no chord is played
    #might need to change threshold value
    id_chord[np.where(max_cor < 0.8*np.max(max_cor))] = 0
    print('Time (s)', 'Chord')
    for n in range(nFrames):
    	print(timestamp[n], CHORDS[id_chord[n]])

    #csv output
    with open(os.path.join(RESULT_DIRECTORY, FILEBASENAME + "_chords.csv"), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for n in range(nFrames):
            writer.writerow([timestamp[n], CHORDS[id_chord[n]]])
    
    #Plotting all figures
    #plt.figure(1)
    #NOTES = ['G','G#','A','A#','B','C','C#','D','D#','E','F','F#']
    #plt.xticks(np.arange(12), NOTES)
    #plt.title('Pitch Class Profile')
    #plt.xlabel('Note')
    #plt.grid(True)
    #
    #plt.figure(2)
    #plt.yticks(np.arange(25), CHORDS)
    #plt.plot(timestamp, id_chord)
    #plt.xlabel('Time in seconds')
    #plt.ylabel('Chords')
    #plt.title('Identified chords')
    #plt.grid(True)
    #plt.show()
