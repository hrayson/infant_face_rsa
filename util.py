import numpy as np
from scipy.io import loadmat

def read_set_events(filename, epoch_evt='mov1', ignore_fields=None):
    EEG = loadmat(filename, uint16_codec='latin1', struct_as_record=False, squeeze_me=True)

    if 'event' in EEG:
        actors = [event.actor.capitalize() for event in EEG['event'] if event.type == epoch_evt]

        movements = [event.movement for event in EEG['event'] if event.type == epoch_evt]
        for i in range(len(movements)):
            if movements[i] == 'smil':
                movements[i] = 'Joy'
            elif movements[i] == 'mopn':
                movements[i] = 'MouthOpening'
            elif movements[i] == 'frwn':
                movements[i] = 'Sadness'

        statuses = [event.code for event in EEG['event'] if event.type == epoch_evt]
        for i in range(len(statuses)):
            if statuses[i] == 'shuf':
                statuses[i] = 'shuffled'
            else:
                statuses[i] = 'unshuffled'

        return np.array(actors), np.array(movements), np.array(statuses)
    else:
        return np.array([]), np.array([]), np.array([])