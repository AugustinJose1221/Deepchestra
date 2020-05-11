#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:30:15 2020

@author: augustinjose
"""


import os
from music21 import converter, pitch, interval, stream
import numpy as np
import tensorflow as tf
# Define save directory
save_dir = '../MIDI/Beethoven/'

# Identify list of MIDI files
songList = os.listdir(save_dir)

# Create empty list for scores
originalScores = []

# Load and make list of stream objects
for song in songList:
    score = converter.parse(save_dir+song)
    originalScores.append(score)
    
from music21 import instrument

# Define function to test whether stream is monotonic
def monophonic(stream):
    try:
        length = len(instrument.partitionByInstrument(stream).parts)
    except:
        length = 0
    return length == 1
print(0)
# Merge notes into chords
originalScores = [song.chordify() for song in originalScores]

from music21 import note, chord

# Define empty lists of lists
originalChords = [[] for _ in originalScores]
originalDurations = [[] for _ in originalScores]
originalKeys = []
print(1)
# Extract notes, chords, durations, and keys
for i, song in enumerate(originalScores):
    originalKeys.append(str(song.analyze('key')))
    for element in song:
        if isinstance(element, note.Note):
            originalChords[i].append(element.pitch)
            originalDurations[i].append(element.duration.quarterLength)
        elif isinstance(element, chord.Chord):
            originalChords[i].append('.'.join(str(n) for n in element.pitches))
            originalDurations[i].append(element.duration.quarterLength)
    #print(str(i))
    
print(2)    
cMajorChords = [c for (c, k) in zip(originalChords, originalKeys) if (k != 'C major')]
cMajorDurations = [c for (c, k) in zip(originalDurations, originalKeys) if (k != 'C major')]


uniqueChords = np.unique([i for s in originalChords for i in s])
#print(uniqueChords)
chordToInt = dict(zip(uniqueChords, list(range(0, len(uniqueChords)))))

# Map unique durations to integers
uniqueDurations = np.unique([i for s in originalDurations for i in s])
durationToInt = dict(zip(uniqueDurations, list(range(0, len(uniqueDurations)))))

intToChord = {i: c for c, i in chordToInt.items()}
intToDuration = {i: c for c, i in durationToInt.items()}



sequenceLength = 320

# Define empty arrays for train data
trainChords = []
trainDurations = []

# Construct training sequences for chords and durations
for s in range(len(cMajorChords)):
    chordList = [chordToInt[c] for c in cMajorChords[s]]
    durationList = [durationToInt[d] for d in cMajorDurations[s]]
    for i in range(len(chordList) - sequenceLength):
        trainChords.append(chordList[i:i+sequenceLength])
        trainDurations.append(durationList[i:i+sequenceLength])



print(3)
print(trainChords)

trainChords = tf.keras.utils.to_categorical(trainChords).transpose(0,2,1)

# Convert data to numpy array of type float
trainChords = np.array(trainChords, np.float)

nSamples = trainChords.shape[0]
print(nSamples)
nChords = trainChords.shape[1]
inputDim = nChords * sequenceLength

# Set number of latent features
latentDim = 2
# Flatten sequence of chords into single dimension
trainChordsFlat = trainChords.reshape([2842,18400],order='C')#, nChordsSequence)



encoderInput = tf.keras.layers.Input(shape = (inputDim))

# Define decoder input shape
latent = tf.keras.layers.Input(shape = (latentDim))

# Define dense encoding layer connecting input to latent vector
encoded = tf.keras.layers.Dense(latentDim, activation = 'tanh')(encoderInput)

# Define dense decoding layer connecting latent vector to output
decoded = tf.keras.layers.Dense(inputDim, activation = 'sigmoid')(latent)

# Define the encoder and decoder models
encoder = tf.keras.Model(encoderInput, encoded)
decoder = tf.keras.Model(latent, decoded)

# Define autoencoder model
autoencoder = tf.keras.Model(encoderInput, decoder(encoded))

autoencoder.compile(loss='binary_crossentropy', optimizer='rmsprop')

# Train autoencoder
autoencoder.fit(trainChordsFlat, trainChordsFlat, epochs = 500)
print(4)

generatedChords = decoder(np.random.normal(size=(1,latentDim))).numpy().reshape(nChords, sequenceLength).argmax(0)

chordSequence = [intToChord[c] for c in generatedChords]


generated_dir = '../Output/'

# Generate stream with guitar as instrument
generatedStream = stream.Stream()

generatedStream.append(instrument.Guitar())
print(5)
# Append notes and chords to stream object
for j in range(len(chordSequence)):
    try:
        generatedStream.append(note.Note(chordSequence[j].replace('.', ' ')))
    except:
        generatedStream.append(chord.Chord(chordSequence[j].replace('.', ' ')))

generatedStream.write('midi', fp=generated_dir+'Beethoven.mid') 
