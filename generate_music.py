import pickle
import random
import numpy as np

import torch
import torch.nn as nn

from music21 import stream, note, chord, instrument

with open("pitchnames.pkl", "rb") as f:
    pitchnames = pickle.load(f)

with open("note_to_int.pkl", "rb") as f:
    note_to_int = pickle.load(f)

int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

network_input = np.load("network_input.npy")

n_vocab = len(pitchnames)

class MusicLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super(MusicLSTM, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=256,
            batch_first=True
        )

        self.dropout1 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(256, 128)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):

        output, (hidden, cell) = self.lstm1(x)

        hidden = self.dropout1(hidden[-1])

        output = self.fc1(hidden)

        output = self.relu(output)

        output = self.fc2(output)

        return output

model = MusicLSTM(
    input_size=1,
    hidden_size=256,
    output_size=n_vocab
)

model.load_state_dict(torch.load("music_model.pth"))

model.eval()

print("Advanced AI Music Generator Ready.")

start = random.randint(0, len(network_input)-1)

pattern = network_input[start]

prediction_output = []

def sample_with_temperature(predictions, temperature=0.9):

    predictions = predictions.detach().numpy()[0]

    predictions = np.log(predictions + 1e-8) / temperature

    exp_preds = np.exp(predictions)

    predictions = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, predictions, 1)

    return np.argmax(probas)

for note_index in range(350):

    prediction_input = torch.tensor(
        pattern,
        dtype=torch.float32
    ).unsqueeze(0)

    prediction = model(prediction_input)

    prediction = torch.softmax(prediction, dim=1)

    index = sample_with_temperature(prediction)

    result = int_to_note[index]

    prediction_output.append(result)

    pattern = np.append(
        pattern[1:],
        [[index / float(n_vocab)]],
        axis=0
    )

print("Music generation complete.")

offset = 0

output_notes = []

for pattern in prediction_output:

    if ('.' in pattern) or pattern.isdigit():

        notes_in_chord = pattern.split('.')

        notes = []

        for current_note in notes_in_chord:

            new_note = note.Note(int(current_note))

            new_note.storedInstrument = instrument.Piano()

            notes.append(new_note)

        new_chord = chord.Chord(notes)

        new_chord.offset = offset

        output_notes.append(new_chord)

    else:

        new_note = note.Note(pattern)

        new_note.offset = offset

        new_note.storedInstrument = instrument.Piano()

        output_notes.append(new_note)

    offset += 0.35

midi_stream = stream.Stream(output_notes)

midi_stream.write('midi', fp='generated_music.mid')

print("Generated advanced MIDI saved as generated_music.mid")
