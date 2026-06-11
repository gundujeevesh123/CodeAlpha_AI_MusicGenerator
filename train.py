import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

network_input = np.load("network_input.npy")
network_output = np.load("network_output.npy")

with open("pitchnames.pkl", "rb") as f:
    pitchnames = pickle.load(f)

n_vocab = len(pitchnames)

X = torch.tensor(network_input, dtype=torch.float32)
y = torch.tensor(network_output, dtype=torch.long)

dataset = TensorDataset(X, y)

loader = DataLoader(dataset, batch_size=64, shuffle=True)

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

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 60

print("Starting advanced AI music training...")

for epoch in range(epochs):

    total_loss = 0

    for batch_x, batch_y in loader:

        optimizer.zero_grad()

        outputs = model(batch_x)

        loss = criterion(outputs, batch_y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "music_model.pth")

print("Advanced AI Music Model Trained Successfully.")
