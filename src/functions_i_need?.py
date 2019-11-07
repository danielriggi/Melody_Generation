
def get_lead(midi, val):
    for instrument in midi.instruments:
        if instrument.name == val:
            return instrument

def leads_at_zero(corpus):
    lst = []
    for song in corpus:
        pm = pretty_midi.PrettyMIDI(song)
        if pm.instruments[0].name in melody_list:
            lst.append(song)
    return lst
            
def write_MIDI_from_list(midi_list):
    for midi in midi_list:
        temp_midi = pretty_midi.PrettyMIDI(midi)
        just_lead = pretty_midi.PrettyMIDI()
        lead = temp_midi.instruments[0]
        just_lead.instruments.append(lead)
#         remove_opening_rest(just_lead)
#         remove_internal_rest(just_lead)
        just_lead.write(midi.replace('data/originals', 'data/leads_with_rests'))

def write_MIDI_from_dict(midi_dict):
    for key, val in mel_per_song.items():
        this_midi = pretty_midi.PrettyMIDI(key)
        this_lead = pretty_midi.PrettyMIDI()
        this_inst = get_lead(this_midi, val)
        this_lead.instruments.append(this_inst)
#         remove_opening_rest(this_lead)
#         remove_internal_rest(this_lead)
        this_lead.write(key.replace('data/originals', 'data/leads_with_rests'))

def get_list_pretty_mid(folder):
    lst = []
    for file in glob.glob(folder):
        mid = pretty_midi.PrettyMIDI(file)
        lst.append(mid)
    return lst

def make_labels(corpus):
    lst = []
    for song in corpus:
        lst.append(np.unique(song))
    flat_list = [item for sublist in lst for item in sublist]
    return sorted(list(np.unique(flat_list)))

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

def check_times_signature(midi_list):
    lst = []
    for midi in midi_list:
        song = converter.parse(midi)
        ts = song[0][2]
        lst.append(ts.ratioString)
    return lst    

def check_bpm(midi_list):
    lst = []
    for midi in midi_list:
        song = converter.parse(midi)
        bpm = song[0][1]
        lst.append(bpm.getQuarterBPM())
    return lst



    # for f in corpus:
    #     fg = converter.parse(f)
    #     print(fg[0][1].number)

    #lst=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    # lst_iter = iter(lst)
    # idx = 0
    # for val in lst_iter:
    #     print(idx,val)
    #     idx += 1
    #     if val == 3:
    #         next(islice(lst_iter,2,3))
    #         idx += 3 



        # if check_for_chord(one_hot_song, idx):
        #     ch = chord.Chord([get_chord(one_hot_song, idx)])
        #     part.append(ch)
        #     step += get_note_count_from_chord(one_hot_song, idx)
        #     idx += step+1
        #     next(islice(song_iter,step,step+1))
        # if 128 <= val <= 255:
        #     pass
        # if 256 <= val <= 356:
        #     time_in_seconds = (val-255)/100
        #     quarter_length = temp.secondsToDuration(time_in_seconds).quarterLength
        #     if idx == 0: #if first element in array represents a time-shift, it is a rest
        #         r = note.Rest(quarterLength=quarterLength)
        #         part.append(r)
        #     elif check_if_rest(one_hot_encoded_song, idx): #check if time shift represents rest
        #         r = note.Rest(quarterLength=quarterLength)
        #         part.append(r)
        #     else: #add time shift to last note
        #         note = part[idx-2]
                
        # if 356 <= val <= 387:
        #     note = part[idx-1]
        #     note.volume.velocity = (val-355)*4-1
    
# for filename in os.listdir('data/corpus'):
#     f = filename.replace('nonstop2k.com','').replace('feat','ft.').replace(' ','_').replace('(Original Mix)','').replace('(midi by Carlo Prato)', '').replace('(www.cprato.com)','')
#     f = f.replace('(midi_by_Carlo_Prato)_', '').replace('_(Original_Mix)','').replace('..','')
#     f = f.replace('-20190812190453-','').replace('-20190522144405-','').replace('-20190906104700-','').replace('-20190702210618-','')
#     f = f.replace('-20190716205438-','').replace('-20171030170239-','')
#     f = f.replace('-theseus','').replace('(Midi_by_carlo_Prato)_','').replace('-andrew-ushakov96-','')
#     f = f.replace('_-_','_').replace('-','_').replace('_.mid','.mid').replace('__','_').replace('-.mid','.mid')
#     src = 'data/corpus/' + filename
#     dst = 'data/corpus/' + f
#     os.rename(src,dst)

# start = np.random.randint(0, len(x)-1)
# int_to_note = dict((number, note) for number, note in enumerate(labels))
# pattern = x[start]
# prediction_output = []
# # generate 500 notes
# for note_index in range(500):
#     prediction_input = np.reshape(pattern, (1, len(pattern), 1))   
#     prediction = model.predict(prediction_input, verbose=0)    
#     index = np.argmax(prediction)
#     result = int_to_note[index]
#     #print(result)
#     prediction_output.append(result) 
#     np.append(pattern, index))
#     pattern = pattern[0:len(pattern)]