import pretty_midi
from music21 import *
import numpy as np
import glob
import itertools
import _pickle as pickle



def get_files_list(folder='data/leads/*.mid'):
    lst = []
    for file in glob.glob(folder):
        lst.append(file)
    return lst

def get_notes_from_chord(m21_chord):
    lst = [p.midi for p in m21_chord.pitches]
    return lst

def get_vel_from_chord(m21_chord):
    vel = m21_chord.volume.velocity
    return vel   

def get_pitch_from_note(m21_note):
    note = m21_note.pitch.midi
    return note  

def one_hot_note_on(midi_note):
    '''Given an integer between 1 and 128 representing a midi note,
    one-hot-encode the note on event in an array of length 388.
    Note on events are represented in the first 128 indices.
    
    Example:
        one_hot_note_on(10)
    Returns:
        array([0,0,0,0,0,0,0,0,0,1,0....0])'''
    arr = np.zeros(388) 
    arr[midi_note-1] = 1 #subtract 1 for zero-indexing 
    return arr

def one_hot_note_off(midi_note):
    '''Given an integer between 1 and 128 representing a midi note,
    one-hot-encode the note on event in an array of length 388.
    Note off events are indices 128-255.'''
    
    arr = np.zeros(388)
    arr[128+midi_note-1] = 1 #first 128 elements are note-on, subtract 1 for zero-indexing 
    return arr  

def one_hot_time(time_in_seconds):
    '''Takes a time in milliseconds and one-hot-encodes the time portion of an array of length 388, 
    the first 256 elements are to encode note on and note off events. 
    The next 100 are for time (what this function encodes), and the final 32 are for velocity.
    Each element of the array represents an incrementing period of 10ms, rounded to the nearest 10ms. 
    
    Example:
        one_hot_time(26) 
    Returns:
        array([...0,0,0,1,0,0...]) ----> Where 1 is at index 259'''
    time_in_milliseconds = time_in_seconds * 1000
    arr = np.zeros(388)
    idx = time_in_milliseconds//10 + 256 #first 256 elements of array are for note events
    if time_in_milliseconds%10 >= 5:
        idx += 1
    arr[int(idx)] = 1
    return arr

def one_hot_vel(velocity):
    arr = np.zeros(388)
    idx = velocity//4
    arr[idx+355] = 1 #velocity is represented at indices 355-387
    return arr

def one_hot_song(midi):
    
    '''One hot encode an entire MIDI. Each event in a MIDI is one hot encoded into an 
    array of length 388.
    
    Returns:
        list of one hot encoded arrays of length 388'''
    
    song = converter.parse(midi)
    seconds_map = song.flat.getElementsByClass(['Note', 'Rest', 'Chord']).secondsMap
    one_hot_list = []
    for note in seconds_map:
        note_length = round(note['durationSeconds'],3)
        note_type = note['element']
        if note_type.isRest:
            if note_length > 3.5: #check if time shift is greater than three and a half seconds 
                note_length = 3.0
            while note_length > 0.0:
                if note_length <= 1.0: #if note length is greater than the maximum encodable time shift
                    one_hot_list.append(one_hot_time(note_length))
                    note_length = 0.0
                elif note_length > 1.0:
                    one_hot_list.append(one_hot_time(1.0))
                    note_length -= 1.0
                
                    
        elif note_type.isChord: #check if element is a chord
            vel = get_vel_from_chord(note_type)
            note_lst = get_notes_from_chord(note_type) #if element is a chord, get all notes
            for note in note_lst: #one hot each note-on in the chord
                one_hot_list.append(one_hot_note_on(note)) 
            one_hot_list.append(one_hot_vel(vel)) #one hot velocity
            while note_length > 0.0:
                if note_length <= 1.0: #if note length is greater than the maximum encodable time shift
                    one_hot_list.append(one_hot_time(note_length))
                    note_length = 0.0
                elif note_length > 1.0:
                    one_hot_list.append(one_hot_time(1.0))
                    note_length -= 1.0            
            for note in note_lst:
                one_hot_list.append(one_hot_note_off(note)) #turn off each note in chord

        else:
            vel = note_type.volume.velocity #get velocity from note
            pitch = get_pitch_from_note(note['element'])
            one_hot_list.append(one_hot_note_on(pitch)) #one hot note-on
            one_hot_list.append(one_hot_vel(vel))
            if note_length <= 1.0: #if note length is greater than the maximum encodable time shift
                    one_hot_list.append(one_hot_time(note_length))
                    note_length = 0.0
            elif note_length > 1.0:
                    one_hot_list.append(one_hot_time(1.0))
                    note_length -= 1.0 
            one_hot_list.append(one_hot_note_off(pitch))  
    return one_hot_list





def to_categorical(y, num_classes=None, dtype='float32'): #from source code (keras/np_utils)
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) +1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical



def prepare_seq(notes, n_vocab, seq_length=32):
    pitchnames = make_labels(notes)
    note_to_int = dict((note, number + 1) for number, note in enumerate(pitchnames))
    #print(note_to_int)
    net_input = []
    net_output = []
    for i in range(0, len(notes) - seq_length, 1):
        seq_in = notes[i:i + seq_length]
        seq_out = notes[i + seq_length]
        net_input.append([note_to_int[char] for char in seq_in])
        net_output.append(note_to_int[seq_out])
        
    n_patterns = len(net_input)
    # reshape to compatible LSTM format
    net_input = np.reshape(net_input, (n_patterns, seq_length, 1))
    # normalize input
    net_input = net_input/float(n_vocab)
    net_output = to_categorical(net_output)
    return (net_input, net_output)





    

if __name__ == '__main__':

    melody_list = ['Lead', 'Spår 1', 'Voice', 'Right hand', 'Melody', 'Blur Lead', '']
    corpus = get_files_list(folder='data/leads_with_rests/*.mid')
    with open('leads_at_index_zero.pkl', 'rb') as f_open:
        leads_at_index_zero = pickle.load(f_open)  
    with open('song_dictionary.pkl', 'rb') as f_op:
        melody_dictionary = pickle.load(f_op)

    