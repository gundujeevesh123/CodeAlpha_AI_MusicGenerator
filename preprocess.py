from music21 import converter, instrument, note, chord
import glob
import pickle

notes = []

midi_files = glob.glob("dataset/classical/*.mid")

print(f"Found {len(midi_files)} MIDI files.")

for file in midi_files:

    print(f"Processing: {file}")

    midi = converter.parse(file)

    parts = instrument.partitionByInstrument(midi)

    if parts:
        notes_to_parse = parts.parts[0].recurse()
    else:
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:

        if isinstance(element, note.Note):
            notes.append(str(element.pitch))

        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

print(f"Total notes extracted: {len(notes)}")

with open("notes.pkl", "wb") as filepath:
    pickle.dump(notes, filepath)

print("Notes saved successfully.")
