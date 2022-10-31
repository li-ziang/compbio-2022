import torch
import torch.nn as nn


class Net_encoder(nn.Module):
    def __init__(self, config):
        super(Net_encoder, self).__init__()
        self.input_size = config.input_size
        self.k = 64
        self.f = 64
        
        if config.encoder_act:
            hidden = 2048
            self.encoder = nn.Sequential(
                nn.Dropout(config.encoder_drop_rate),
                nn.LayerNorm(self.input_size),
                nn.Linear(self.input_size, hidden),
                nn.ReLU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, 64),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_size, 64),
            )
        self.space_alignment = config.space_alignment
        if config.space_alignment:
            self.log_scale = nn.Parameter(torch.zeros(1, 64).float())
            self.trans = nn.Parameter(torch.zeros(1, 64).float())
            

    def forward(self, data, rna_data=True):
        data = data.float().view(-1, self.input_size)
        embedding = self.encoder(data)
        if self.space_alignment and not rna_data:
            embedding = torch.exp(self.log_scale) * embedding + self.trans
            # with torch.no_grad():
            #     self.log_scale += torch.randn_like(self.log_scale) * 0.005
            #     self.trans += torch.randn_like(self.trans) * 0.005
        return embedding


class Net_cell(nn.Module):
    def __init__(self, config):
        super(Net_cell, self).__init__()
        num_of_class = config.number_of_class
        self.cell = nn.Sequential(
            nn.Linear(64, num_of_class)
        )

    def forward(self, embedding):
        cell_prediction = self.cell(embedding)
        return cell_prediction
