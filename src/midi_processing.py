from music21 import *
import numpy as np
import glob
import os
import re
from itertools import islice
import _pickle as pickle



def get_files_list(folder='../data/leads/*.mid'):
    lst = []
    for file in glob.glob(folder):
        lst.append(file)
    return lst


def set_time(m21_song):
    mm = tempo.MetronomeMark(number=150)
    ts = meter.TimeSignature('4/4')
    ts_iter = m21_song.getElementsByClass(meter.TimeSignature)
    mm_iter = m21_song.getElementsByClass(tempo.MetronomeMark)
    if len(ts_iter) == 0:
        m21_song.insert(0,ts)
    else:
        for val in ts_iter:
            m21_song.pop(m21_song.index(val))
        m21_song.insert(0,ts)
    if len(mm_iter) == 0:
        m21_song.insert(0,mm)
    else:
        for val in mm_iter:
            m21_song.pop(m21_song.index(val))
        m21_song.insert(0,mm)

def get_longrests_idxs(song):
    lst = []
    for rest in song.getElementsByClass("Rest"):
        if rest.duration.quarterLength > 1.5:
            lst.append(song.index(rest))
    for note in song.getElementsByClass(['Note', 'Chord']):
        idx = song.index(note)
        if idx+1 == len(song):
            continue
        else:
            nxt = song[idx+1]
            prev = song[idx-1]
        if isinstance(prev, meter.TimeSignature) and note.offset != 0.0:
            lst.append(idx-1)
        elif nxt.offset > (note.offset + (float(note.duration.quarterLength)*5)): #*5 is arbitrary
            lst.append(idx)
    return lst 

def shorten_rest(long_rest_index, song):
    r = song[long_rest_index]
    nxt = song[long_rest_index+1]
    ql = r.duration.quarterLength
    new_ql = 0.5
    if isinstance(r,note.Rest) and nxt.offset < float(r.duration.quarterLength): #if rest exists while other note is playing, pass
        pass
    elif isinstance(r, meter.TimeSignature): #for songs where an opening rest isn't encoded(first element is a note with an offset other than zero)
        os_diff = nxt.offset
        for thing in song[long_rest_index+1:]: #reset the offsets of the remaining elements
            os = thing.offset - os_diff + 0.5 #0.5 added so it is not at same offset as last note 
            song.setElementOffset(thing, os)
    elif isinstance(r,note.Note)  or isinstance(r,chord.Chord):
        os_diff = nxt.offset - r.offset
        for thing in song[long_rest_index+1:]: #reset the offsets of the remaining elements
            os = thing.offset - os_diff + 0.5 #0.5 added so it is not at same offset as last note 
            song.setElementOffset(thing, os)
    else:
        r.duration.quarterLength = new_ql        
        for thing in song[long_rest_index+1:]: #reset the offsets of the remaining elements
            os = thing.offset - (ql-new_ql) 
            song.setElementOffset(thing, os)

def shorten_all_rests(indx_lst, song):
    for idx in indx_lst:
        shorten_rest(idx, song)

def remove_trailing_inst(song):
    '''remove the instruments that appear at the end of songs'''
    lst = []
    for inst in reversed(song):
        if isinstance(inst, instrument.Instrument):
            i = song.pop(song.index(inst))
            lst.append(i)
        else:
            break
    return lst

def get_original_notes(song, trailing_inst_list):
    '''Given a music21 song, get just the notes, chords, and rests.'''
    for thing in song.getElementsByClass(meter.TimeSignature): #get first timesignature
        ts = thing
        idx = song.index(ts)
        break
    song_length = len(song)
    ncr = song[idx+1:song_length-len(trailing_inst_list)]#get the song, minus the instruments at the beginning and end, and timesignatures
    return ncr

def double_notes_length(song, notes):
    '''Given the original notes, rests, and chords (notes) of the song, add them to the end of the current version of the song'''
    offset = notes[-1].offset#get the offset of the last instrument
    for n in notes:
        if n.isNote:
            new_n = note.Note(n.pitch)
            new_n.volume = n.volume
            new_n.duration.quarterLength = n.duration.quarterLength
            new_n.offset = n.offset + offset 
            song.append(new_n)
        elif n.isChord:
            new_ch = chord.Chord()
            new_ch.pitches = n.pitches
            new_ch.volume = n.volume
            new_ch.duration.quarterLength = n.duration.quarterLength
            new_ch.offset = n.offset + offset 
            song.append(new_ch)
        else:
            new_r = note.Rest()
            new_r.offset = n.offset + offset 
            new_r.duration.quarterLength = n.duration.quarterLength
            song.append(new_r)

def write_to_mid(song_name, song, len_longest_song=951):
    if len(song)< len_longest_song*0.75:
        print(song_name)
        inst_lst = remove_trailing_inst(song)
        orig_ncr = get_original_notes(song,inst_lst)
        while len(song) < len_longest_song*0.75: #if less than 75% of the length of the longest song, add original notes to end of song
            double_notes_length(song, orig_ncr)
        song.append(inst_lst)
    lst = get_longrests_idxs(song)            
    shorten_all_rests(lst, song)
    mf = midi.translate.streamToMidiFile(song)
    mf.open(song_name.replace('full_songs','leads_doubled'), 'wb')
    mf.write()
    mf.close() 

def extract_zero_indexed_lead(leads_at_zero_index):
    for song in leads_at_zero_index: 
        s = converter.parse(song)
        part = s[0]
        set_time(part)
        if part.hasVoices():
            part.flattenUnnecessaryVoices(force=True,inPlace=True)
            write_to_mid(song, part)
        else:
            write_to_mid(song, part)

def extract_nonzero_indexed_leads(mel_dict):
    for key, val in mel_dict.items():
        s = converter.parse(key)
        p = s[val]
        set_time(p)
        if p.hasVoices():
            p.flattenUnnecessaryVoices(force=True,inPlace=True)
            write_to_mid(key, p)
        else:
            write_to_mid(key,p)

def get_longest(corpus):
    lst = []
    for song in corpus:
        s = converter.parse(song)
        l = len(s[0])
        lst.append(l)
    return max(lst)

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
    if time_in_milliseconds == 1000.0:
        idx = 355
    else:
        idx = time_in_milliseconds//10 + 256
        if time_in_milliseconds%10 >= 5:
            idx += 1        
    arr[int(idx)] = 1
    return arr

def one_hot_vel(velocity):
    arr = np.zeros(388)
    idx = velocity//4
    arr[idx+356] = 1 #velocity is represented at indices 356-387
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

def is_rest(one_hot_song, event_index):
    '''Given a one hot encoded song and an index representing a time shift,
    return True if rest, False otherwise.'''
    idxs = np.nonzero(one_hot_song)
    index_to_check = event_index - 1
    event = idxs[1][index_to_check]
    if event >= 356:
        return False
    elif 128 <= event <= 255:
        return True
    else:
        return is_rest(one_hot_song, event_index=index_to_check)
    
def check_for_chord(one_hot_song, note_index):
    this_event = np.nonzero(one_hot_song)[1][note_index]
    event_to_check = np.nonzero(one_hot_song)[1][note_index+1]
    if 0 <= this_event <= 127 and 0 <= event_to_check <= 127:
        return True
    else:
        return False

def get_note_count_chord(one_hot_song, first_note_index):
    count = 1
    index = first_note_index
    while check_for_chord(one_hot_song, index):
        count += 1
        index += 1
    return count

def get_chord(one_hot_song, first_note_index):
    lst = []
    note_count = get_note_count_chord(one_hot_song, first_note_index)
    index = first_note_index
    for num in range(note_count):
        note = np.nonzero(one_hot_song)[1][index] 
        lst.append(note)
        index += 1
    return lst

def check_next(one_hot_song, index):
    val = np.nonzero(one_hot_song)[1][index+1]
    return val

def get_ts_duration(one_hot_song, time_shift_index):
    dur = 0.0
    idx = time_shift_index 
    dur += ((np.nonzero(one_hot_song)[1][idx])-255)/100
    count = 1
    while 256 <= check_next(one_hot_song, idx) <= 355:
        val = np.nonzero(one_hot_song)[1][idx+1]
        time_in_seconds = (val-255)/100
        dur += time_in_seconds
        idx += 1
        count += 1
    return dur, count

def rename_files(directory='../data/full_songs/'):
    for filename in os.listdir(directory):
        f = filename.replace('-nonstop2k.com','_').replace('feat','ft').replace('(Original Mix)','').replace('  (midi by Carlo Prato) (www.cprato.com)', '_')
        f = f.replace('-theseus-', '_').replace('(Midi by Carlo Prato) (www.cprato.com)', '_').replace('-rlc-winston-', '_').replace('-andrew-ushakov96-',"_").replace('-','_').replace(' ','_')
        f = re.sub('\d+','_',f)
        f = re.sub('_+','_', f)
        src = directory + filename
        dst = directory + f
        os.rename(src,dst)

def one_hot_to_midi(one_hot_song, inst='Saxophone'):
    #events = np.nonzero(one_hot_encoded_song)[0]
    events = np.nonzero(one_hot_song)[1]
    song_iter = enumerate(events) 
    this_stream = stream.Stream()
    part = stream.Part()
    inst = instrument.Instrument(inst)
    temp = tempo.MetronomeMark('animato')
    met = meter.TimeSignature('4/4')
    this_stream.append(part)
    part.append(inst)
    part.append(temp)
    part.append(met)
    for idx, val in song_iter:
        #print(f'index = {idx} value = {val}')
        if idx == 0 and 256 <= val <= 355:
            time, count = get_ts_duration(one_hot_song, idx)
            if time <= 0.0:
                time = 0.05
            dur = temp.secondsToDuration(time)
            r = note.Rest(duration=dur)
            part.append(r)
            [next(song_iter) for _ in range(count-1)]
        elif idx == len(events)-1 and 256 <= val <= 355:
            pass
        elif check_for_chord(one_hot_song, idx):
            chord_length = get_note_count_chord(one_hot_song,idx)
            vel_index = idx + chord_length
            ts_index = vel_index + 1
            time, count = get_ts_duration(one_hot_song, ts_index)
            if time <= 0.0:
                 time = 0.05
            dur = temp.secondsToDuration(time)
            vel = (events[vel_index] - 355) * 4 - 1
            chord_notes = get_chord(one_hot_song, idx)
            ch = chord.Chord(duration=dur)
            for num in chord_notes:
                p = pitch.Pitch(num)
                string = p.nameWithOctave
                ch.add(string)                
            ch.volume = volume.Volume(velocity=vel)
            part.append(ch)
            step = chord_length*2 + count  # shift the chord length * 2 for note and note off, 1 for velocity, and count for number of time-shifts
            if (idx + step) > len(events)-1:
                break  
            [next(song_iter) for _ in range(step)] 
        elif 256 <= val <= 355 and is_rest(one_hot_song, idx) and idx != 0:
            time, count = get_ts_duration(one_hot_song, idx)
            dur = temp.secondsToDuration(time)
            r = note.Rest(duration=dur)
            part.append(r)
            [next(song_iter) for _ in range(count-1)]
        else:
            time, count = get_ts_duration(one_hot_song, idx+2)
            if time <= 0.0:
                time = 0.05
            vel = (events[idx+1] - 355) * 4 - 1
            dur = temp.secondsToDuration(time)
            n = note.Note(val,duration=dur)
            n.volume.velocity = vel
            part.append(n)
            [next(song_iter) for _ in range(count + 2)] #add two for note off event and velocity
    return this_stream

def transpose_corpus(corpus):
    '''Transpose each midi up and down five semitones, and write to file. '''
    for song in corpus:
        s = converter.parse(song)
        for num in range(1,6):           
            t_down = s.transpose(-num)
            t_up = s.transpose(num)        
            mf_up = midi.translate.streamToMidiFile(t_up)
            mf_down = midi.translate.streamToMidiFile(t_down)
            mf_up.open(song.replace('.mid', f'_transpose_down_{num}.mid'),'wb')
            mf_up.write()
            mf_up.close()
            mf_down.open(song.replace('.mid',f'_transpose_up_{num}.mid'),'wb')
            mf_down.write()
            mf_down.close()

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

def compile_corpus(corpus):
    oh_corp = []
    for song in corpus:
        x = one_hot_song(song)
        oh_corp += x
    return oh_corp

def oh_arr_to_int(oh_arr):
    return np.nonzero(oh_arr)[0][0] 

def prepare_seq(oh_corp, seq_length=100):
    net_input = []
    net_output = []
    for i in range(0, len(oh_corp) - seq_length, 1):
        seq_in = oh_corp[i:i + seq_length]
        seq_out = oh_corp[i + seq_length]
        net_input.append([oh_arr_to_int(char) for char in seq_in])
        net_output.append(seq_out)
    n_patterns = len(net_input)
    net_input = np.reshape(np.array(net_input), (n_patterns, seq_length, 1))
    net_input = net_input/388.0
    return net_input, np.array(net_output)

def generate_from_random(unique_notes,seq_len=100):
    generate = np.random.choice(unique_notes, seq_len)
    return generate.reshape(1,seq_len,1)

def generate_notes(generator, model,length=500, seq_length=100):
    oh_lst = []
    test_input = generator
    for i in range(length):
        pred = model.predict(test_input)
        event_index = np.argmax(pred)
        event = round((pred[:,event_index]*388)[0])
        if event == 388:
            event = 387
        arr = np.zeros(388)
        arr[int(event)] = 1
        stand_event = event/388.0
        oh_lst.append(arr)
        test_input = np.append(test_input[0], stand_event)
        test_input = test_input[-50:].reshape(1,50,1)
    return oh_lst




    

if __name__ == '__main__':
    corpus = get_files_list()
    longest = get_longest(corpus)
    corp = compile_corpus(corpus)
    X, y = prepare_seq(corp)
    # with open('leads_at_index_zero.pkl', 'rb') as f_open:
    #     leads_at_index_zero = pickle.load(f_open)  
    # with open('song_dictionary.pkl', 'rb') as f_op:
    #     melody_dictionary = pickle.load(f_op)


    
