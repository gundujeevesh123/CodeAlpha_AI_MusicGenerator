import pickle
import numpy as np

sequence_length = 30

with open("notes.pkl", "rb") as filepath:
    notes = pickle.load(filepath)

print(f"Total notes loaded: {len(notes)}")

pitchnames = sorted(set(notes))

n_vocab = len(pitchnames)

print(f"Unique notes: {n_vocab}")

note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

network_input = []
network_output = []

for i in range(0, len(notes) - sequence_length):

    sequence_in = notes[i:i + sequence_length]

    sequence_out = notes[i + sequence_length]

    network_input.append([note_to_int[n] for n in sequence_in])

    network_output.append(note_to_int[sequence_out])

n_patterns = len(network_input)

print(f"Total training patterns: {n_patterns}")

network_input = np.reshape(
    network_input,
    (n_patterns, sequence_length, 1)
)

network_input = network_input / float(n_vocab)

network_output = np.array(network_output)

np.save("network_input.npy", network_input)
np.save("network_output.npy", network_output)

with open("note_to_int.pkl", "wb") as f:
    pickle.dump(note_to_int, f)

with open("pitchnames.pkl", "wb") as f:
    pickle.dump(pitchnames, f)

print("Sequence preparation completed.")
