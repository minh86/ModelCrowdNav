from torch import nn

class autoencoder(nn.Module):
    def __init__(self, num_human, drop_rate=0.00):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_human * 4, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12))
        self.decoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, num_human * 2))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x