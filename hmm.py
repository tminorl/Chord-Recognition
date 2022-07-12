"""Automatic chord recogniton with HMM, as suggested by Juan P. Bello in
'A mid level representation for harmonic content in music signals'
@author ORCHISAMA"""

from __future__ import division
from chromagram import compute_chroma
import os
from scipy.io.wavfile import read
import numpy as np 
import json

CHORDS = ['G','G#','A','A#','B','C','C#','D','D#','E','F','F#','Gm','G#m','Am','A#m','Bm','Cm','C#m','Dm','D#m','Em','Fm','F#m']
NESTED_COF = ['G','Bm','D','F#m','A','C#m','E','G#m','B','D#m','F#','A#m','C#',"Fm","G#",'Cm','D#','Gm','A#','Dm','F','Am','C','Em']

"""calculates multivariate gaussian matrix from mean and covariance matrices"""
def multivariate_gaussian(x, meu, cov):
	
	det = np.linalg.det(cov)
	val = np.exp(-0.5 * np.dot(np.dot((x-meu).T, np.linalg.inv(cov)), (x-meu)))
	try:
		val /= np.sqrt(((2*np.pi)**12)*det)
	except:
		print('Matrix is not positive, semi-definite')
	if np.isnan(val):
		val = np.finfo(float).eps
	return val


"""initialize the emission, transition and initialisation matrices for HMM in chord recognition
PI - initialisation matrix, #A - transition matrix, #B - observation matrix"""
def initialize(chroma, templates, NESTED_COF):

	"""initialising PI with equal probabilities"""
	PI = np.ones(24)/24

	"""initialising A based on nested circle of fifths"""
	eps = 0.01
	A = np.empty((24,24))
	for chord in CHORDS:
		ind = NESTED_COF.index(chord)
		t = ind
		for i in range(24):
			if t >= 24:
				t = t%24
			A[ind][t] = (abs(12-i)+eps)/(144 + 24*eps)
			t += 1

	
	"""initialising based on tonic triads - Mean matrix; Tonic with dominant - 0.8,
	tonic with mediant 0.6 and mediant-dominant 0.8, non-triad diagonal	elements 
	with 0.2 - covariance matrix"""

	nFrames = np.shape(chroma)[1]
	B = np.zeros((24,nFrames))
	meu_mat = np.zeros((24,12))
	cov_mat = np.zeros((24,12,12))
	meu_mat = np.array(templates)
	offset = 0

	for i in range(24):
		if i == 12:
			offset = 0
		tonic = offset
		if i<12:
			mediant = (tonic + 4)%12
		else:
			mediant = (tonic + 3)%12
		dominant = (tonic+7)%12

		#weighted diagonal
		cov_mat[i,tonic,tonic] = 0.8
		cov_mat[i,mediant,mediant] = 0.6
		cov_mat[i,dominant,dominant] = 0.8

		#off-diagonal - matrix not positive semidefinite, hence determinant is negative
		# for n in [tonic,mediant,dominant]:
		# 	for m in [tonic, mediant, dominant]:
		# 		if (n is tonic and m is mediant) or (n is mediant and m is tonic):
		# 			cov_mat[i,n,m] = 0.6
		# 		else:
		# 			cov_mat[i,n,m] = 0.8

		#filling non zero diagonals
		for j in range(12):
			if cov_mat[i,j,j] == 0:
				cov_mat[i,j,j] = 0.2
		offset += 1
	

	"""observation matrix B is a multivariate Gaussian calculated from mean vector and 
	covariance matrix"""

	for m in range(nFrames):
		for n in range(24):
			B[n,m] = multivariate_gaussian(chroma[:,m], meu_mat[n,:],cov_mat[n,:,:])

	return (PI,A,B)
	


"""Viterbi algorithm to find Path with highest probability - dynamic programming"""

def viterbi(PI,A,B):
	(nrow, ncol) = np.shape(B)
	path = np.zeros((nrow, ncol))
	states = np.zeros((nrow,ncol))
	path[:,0] = PI * B[:,0]

	for i in range(1,ncol):
		for j in range(nrow):
			s = [(path[k,i-1] * A[k,j] * B[j,i], k) for k in range(nrow)]
			(prob,state) = max(s)
			path[j,i] = prob
			states[j,i-1] = state
	
	return (path,states)

if __name__ == "__main__":
    import sys
    import csv

    #fft size and windowing size
    NFFT = 8192
    HOP_SIZE = 1024

    RESULT_DIRECTORY = './'
    FILENAME = sys.argv[1]
    FILEBASENAME = os.path.splitext(os.path.basename(FILENAME))[0]
    
    with open('chord_templates.json', 'r') as fp:
    	templates_json = json.load(fp)
    
    templates = []
    
    for chord in CHORDS:
    	templates.append(templates_json[chord])
    
    """read audio and compute chromagram"""
    (fs,s) = read(FILENAME)
    
    #reduce sample rate and convert to mono
    x = s[::4]
    x = x[:,1]
    fs = int(fs/4)
    
    #framing audio 
    nFrames = int(np.round(len(x)/(NFFT-HOP_SIZE)))
    
    #zero padding to make signal length long enough to have nFrames
    x = np.append(x, np.zeros(NFFT))
    xFrame = np.empty((NFFT, nFrames))
    start = 0   
    chroma = np.empty((12,nFrames)) 
    timestamp = np.zeros(nFrames)
    
    #compute PCP
    for n in range(nFrames):
    	xFrame[:,n] = x[start:start+NFFT] 
    	start = start + NFFT - HOP_SIZE 
    	chroma[:,n] = compute_chroma(xFrame[:,n],fs)
    	if  np.all(chroma[:,n] == 0):
    		chroma[:,n] = np.finfo(float).eps
    	else:
    		chroma[:,n] /= np.max(np.absolute(chroma[:,n]))
    	timestamp[n] = n*(NFFT-HOP_SIZE)/fs 
    
    
    #get max probability path from Viterbi algorithm
    (PI,A,B) = initialize(chroma, templates, NESTED_COF)
    (path, states) = viterbi(PI,A,B)
    
    #normalize path
    for i in range(nFrames):
    	path[:,i] /= sum(path[:,i])
    
    #choose most likely chord - with max value in 'path'
    final_chords = []
    indices = np.argmax(path,axis=0)
    final_states = np.zeros(nFrames)
    
    
    #find no chord zone
    set_zero = np.where(np.max(path,axis=0) < 0.3*np.max(path))[0]
    if np.size(set_zero) != 0:
    	indices[set_zero] = -1
    
    #identify chords
    for i in range(nFrames):
    	if indices[i] == -1:
    		final_chords.append('NC')
    	else:
    		final_states[i] = states[indices[i],i]
    		final_chords.append(CHORDS[int(final_states[i])])
    
    print('Time(s)','Chords')
    for i in range(nFrames):
    	print(timestamp[i], final_chords[i])

    #csv output
    with open(os.path.join(RESULT_DIRECTORY, FILEBASENAME + "_chords.csv"), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for i in range(nFrames):
            writer.writerow([timestamp[i], final_chords[i]])
