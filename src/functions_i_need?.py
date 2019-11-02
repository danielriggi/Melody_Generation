
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