import pretty_midi
import numpy as np
import glob
import itertools
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Bidirectional
from sklearn.preprocessing import OneHotEncoder





def get_lead(midi, val):
    for instrument in midi.instruments:
        if instrument.name == val:
            return instrument

def remove_opening_rest(prettymid):
    times = prettymid.instruments[0].get_onsets()
    if times[0] != 0:
        prettymid.adjust_times(times, times - times[0])

def remove_internal_rest(prettymid):
    times = prettymid.instruments[0].get_onsets()
    old_times = times.copy()
    for idx, val in enumerate(times[:-1]):
        rest = times[idx+1] - val
        if rest > 2:
            times[idx+1:] -= rest
    prettymid.adjust_times(prettymid.instruments[0].get_onsets(), times)      

def write_MIDI_from_list(midi_list):
    for midi in midi_list:
        temp_midi = pretty_midi.PrettyMIDI(midi)
        just_lead = pretty_midi.PrettyMIDI()
        lead = temp_midi.instruments[0]
        just_lead.instruments.append(lead)
        remove_opening_rest(just_lead)
        remove_internal_rest(just_lead)
        just_lead.write(midi.replace('data', 'data/leads'))

def write_MIDI_from_dict(midi_dict):
    for key, val in mel_per_song.items():
        this_midi = pretty_midi.PrettyMIDI(key)
        this_lead = pretty_midi.PrettyMIDI()
        this_inst = get_lead(this_midi, val)
        this_lead.instruments.append(this_inst)
        remove_opening_rest(this_lead)
        remove_internal_rest(this_lead)
        this_lead.write(key.replace('data', 'data/leads'))

def get_list_pretty_mid(folder):
    lst = []
    for file in glob.glob(folder):
        mid = pretty_midi.PrettyMIDI(file)
        lst.append(mid)
    return lst

def get_notes(mid):
    return mid.instruments[0].notes

def get_pitch(pretty_midi_note):
    return pretty_midi_note.pitch 

def get_duration(pretty_midi_note):
    start = pretty_midi_note.start
    end = pretty_midi_note.end
    dur = end - start
    return round(dur,3)

def get_vel(pretty_midi_note):
    return pretty_midi_note.velocity

def make_psuedo_song(pretty_mid):
    lst = []
    for note in get_notes(pretty_mid):
        pitch = str(get_pitch(note))
        dur = str(get_duration(note))
        vel = str(get_vel(note))
        note_string = f'{pitch}{dur}{vel}'
        lst.append(note_string)
    return lst  

def get_psuedo_notes_corpus(pretty_midi_list):
    lst = []
    for mid in pretty_midi_list:
        lst.append(make_psuedo_song(mid))
    return list(itertools.chain(*lst))


def make_labels(corpus):
    lst = []
    for song in corpus:
        lst.append(np.unique(song))
    flat_list = [item for sublist in lst for item in sublist]
    return sorted(list(np.unique(flat_list)))

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


def create_model(net_input, n_vocab):
    model = Sequential()
    model.add(
        Bidirectional(
            LSTM(512, return_sequences=True),
            input_shape=(net_input.shape[1], net_input.shape[2]),
        )
    )
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(512)))
    model.add(Dense(n_vocab+1))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    

    return model

def train(net_input, net_output, model, epochs=95):
    model.fit(net_input,
            net_output,
            epochs=epochs,
            batch_size=64)


    

if __name__ == '__main__':

    melody_list = ['Lead', 'Spår 1', 'Voice', 'Right hand', 'Melody', 'Blur Lead', '']
    other_zero_list = ['data/Avicii-Without-You-Merk-Kremont-Remix-rlc-winston-20180119095001-nonstop2k.com.mid',
                    'data/Axwell-Ingrosso-I-Love-You-ft-Kid-Ink-andrew-ushakov96-20190610193109-nonstop2k.com.mid',
                    'data/Kygo-Sasha-Sloan-This-Town-rlc-winston-20171030170239-nonstop2k.com.mid',
                    'data/Skrillex_ScaryMonstersAndNiceSprites.mid',
                    'data/Sebastian Ingrosso & Alesso - Calling.mid',
                    'data/Kygo_firestone.mid']
    mel_per_song = {'data/Said-the-Sky-Mountains-ft-Diamond-Eyes-Skrux-Remix-Anonymous-20180119055128-nonstop2k.com.mid': 'Serum 6',
               'data/Audien-Cecilia-Gault-Higher-theseus-20190209171713-nonstop2k.com.mid': 'Piano',
                'data/Andrew-Rayel-NWYR-The-Melody-theseus-20190522144405-nonstop2k.com.mid': 'Lead',
                'data/Cash-Cash-Call-You-feat-Nasri-of-MAGIC-theseus-20190702210618-nonstop2k.com.mid': 'Guitar Arp',
                'data/Illenium-Free-Fall-andrew-ushakov96-20171007220131-nonstop2k.com.mid': 'Lead',
                'data/Avicii-Sandro-Cavazza-Without-You-BassBringer-20180510190520-nonstop2k.com.mid': 'Guitar',
                'data/Alan_Walker_Faded.mid': 'Lead',
                'data/RL-Grime-Miguel-Julia-Michaels-Light-Me-Up-rlc-winston-20180918222247-nonstop2k.com.mid': 'vocal #6',
                'data/Tiesto-Stargate-Aloe-Blacc-Carry-You-Home-rlc-winston-20171007220141-nonstop2k.com.mid': 'vocal party #12',
                'data/Alesso-Nico-Vinz-I-Wanna-Know-BassBringer-20160613224729-nonstop2k.com.mid': 'VOX Piano',
                'data/R3hab-KSHMR-Islands-max123a-20171205145137-nonstop2k.com.mid': 'lead',
                'data/Marshmello-Chvrches-Here-With-Me-theseus-20190522144255-nonstop2k.com.mid': 'Vocal',
                'data/Cash-Cash-All-My-Love-ft-Conor-Maynard-andrew-ushakov96-20170725222833-nonstop2k.com.mid':'Lead',
                'data/Tiesto-Can-You-Feel-It-ft-John-Christian-theseus-20190522144356-nonstop2k.com.mid': 'Lead 1',
                'data/Kygo-Here-For-You-ft-Ella-Henderson-Frozen-Ray-20160105221803-nonstop2k.com.mid': 'Lead',
                'data/Kaskade-Ilsey-Disarm-You-ft-Ilsey-Frozen-Ray-20150920121437-nonstop2k.com.mid': 'Lead',
                'data/RL-Grime-Shrine-ft-Freya-Ridings-rlc-winston-20180825090537-nonstop2k.com.mid': 'pluck #5',
                'data/Alan-Walker-Darkside-ft-Tomine-Harket-Au-Ra-rlc-winston-20180819174026-nonstop2k.com.mid': 'vocal #15',
                'data/Martin-Garrix-High-On-Life-ft-Bonn-rlc-winston-20180818221000-nonstop2k.com.mid': 'vocal #12',
                'data/R3hab-Quintino-I-Just-Can-t-andrew-ushakov96-20171013110942-nonstop2k.com.mid': 'Pluck 2',
                'data/The-Chainsmokers-Tritonal-Until-You-Were-Gone-andrew-ushakov96-20160609222510-nonstop2k.com.mid': 'Lead',
                'data/Avicii-Imagine-Dragons-Heart-Upon-My-Sleeve-theseus-20190616202406-nonstop2k.com.mid': 'Vocal 1',
                'data/Zedd-Alessia-Cara-Stay-Acoustic-version-rlc-winston-20180409063340-nonstop2k.com.mid': 'vocal #6',
                'data/Dimitri Vegas & Like Mike - Stay A While  (Original Mix) (midi by Carlo Prato) (www.cprato.com).mid': 'Lead',
                'data/Hardwell-Timmy-Trumpet-The-Underground-andrew-ushakov96-20180217050005-nonstop2k.com.mid': 'Lead',
                'data/RL-Grime-Daya-I-Wanna-Know-rlc-winston-20180418200244-nonstop2k.com.mid': 'lead #2'}


    # pitch_corpus = []
    # vel_corpus= []
    # for file in glob.glob("data/leads/*.mid"):
    #     try:
    #         midi_pretty = pretty_midi.PrettyMIDI(file)
    #         for note in midi_pretty.instruments[0].notes:
    #                 pitch_corpus.append(note.pitch)
    #                 vel_corpus.append(note.velocity)
    #     except:
    #         print(f'error with {file}')
    # corp_pitch = list(set(pitch_corpus))
    # corp_vel = list(set(vel_corpus))