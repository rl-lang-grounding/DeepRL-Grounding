import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class A3C_LSTM_GA(torch.nn.Module):

    def __init__(self, args):
        super(A3C_LSTM_GA, self).__init__()

        # Image Processing
        self.conv1 = nn.Conv2d(3, 128, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)

        # Instruction Processing
        self.gru_hidden_size = 256
        self.input_size = args.input_size
        self.embedding = nn.Embedding(self.input_size, 32)
        self.gru = nn.GRU(32, self.gru_hidden_size)

        # Gated-Attention layers
        # self.attn_linear = nn.Linear(self.gru_hidden_size, 64)
        self.attn_linear1 = nn.Linear(self.gru_hidden_size, 64)
        self.attn_linear2 = nn.Linear(self.gru_hidden_size, 64)
        self.attn_linear3 = nn.Linear(self.gru_hidden_size, 64)
        self.attn_linear4 = nn.Linear(self.gru_hidden_size, 64)
        self.attn_linear5 = nn.Linear(self.gru_hidden_size, 64)

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(
                args.max_episode_length+1,
                self.time_emb_dim)

        # A3C-LSTM layers
        self.linear = nn.Linear(5 * 8 * 17, 256)
        self.lstm = nn.LSTMCell(256, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 3)

        # Initializing weights
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()

    def forward(self, inputs):
        x, input_inst, (tx, hx, cx) = inputs

        # Get the image representation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x_image_rep = F.relu(self.conv3(x))

        # Get the instruction representation
        encoder_hidden = Variable(torch.zeros(1, 1, self.gru_hidden_size))
        for i in xrange(input_inst.data.size(1)):
            word_embedding = self.embedding(input_inst[0, i]).unsqueeze(0)
            _, encoder_hidden = self.gru(word_embedding, encoder_hidden)
        x_instr_rep = encoder_hidden.view(encoder_hidden.size(1), -1)

        # Get the attention vector from the instruction representation
        
        attention_layers = [self.attn_linear1,self.attn_linear2,self.attn_linear3,self.attn_linear4,self.attn_linear5]
        for i in range(5) :
            attn_map = F.relu(attention_layers[i](x_instr_rep))
            attn_map = attn_map.view(1,64,1,1)
            attn_vec = F.conv2d(x_image_rep,attn_map)
            if i == 0 :
                attn_vecs = attn_vec
            else :
                attn_vecs = torch.cat((attn_vecs,attn_vec),1)    
        
        x = attn_vecs
        

        x = x.view(x.size(0), -1)

        # A3C-LSTM
        x = F.relu(self.linear(x))
        hx, cx = self.lstm(x, (hx, cx))
        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
