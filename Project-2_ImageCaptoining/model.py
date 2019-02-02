import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        
        ## Added after Project2 is finished
        ## tried: 
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        #with torch.no_grad(): #Added togrther with nn.BatchNorm1d
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        
        ## Added after Project2 is finished
        ## tried: 
        features = self.bn(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0):
        super(DecoderRNN, self).__init__()
        # save params
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        # define layers
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, \
                            hidden_size=hidden_size, \
                            num_layers=num_layers, 
                            bias=True,
                            dropout=0,
                            batch_first=True)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        #self.hidden2out = nn.Linear(hidden_size, vocab_size)
        # self.softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=2)
    
    def init_hidden(self, batch_size):
        ''' Initialize a hidden state at the start of training,
            Hidden state is filled with all zeroes
            The axes semantics are (num_layers, batch_size, hidden_dim) '''
        return (torch.zeros((1, batch_size, self.hidden_size), device=device), \
                torch.zeros((1, batch_size, self.hidden_size), device=device))
        
    def forward(self, features, captions):
        
        # captions = self.word_embeddings(captions[:,:-1])
        captions_without_end = captions[:, :-1]
        # embeddings new shape : (batch_size, captions length - 1, embed_size)
        captions = self.word_embeddings(captions_without_end)  
        
        # Initialize the hidden state
        batch_size = features.shape[0] # shape of features = (batch_size, embed_size)
        self.hidden = self.init_hidden(batch_size) 
                
        # Stack the features and captions
        inputs = torch.cat((features.unsqueeze(1), captions), dim=1)
        lstm_output, self.hidden = self.lstm(inputs, self.hidden)        
        # outputs = self.hidden2out(lstm_out)
        outputs = self.linear(lstm_output)
        return outputs

    def sample(self, inputs, states=None, max_len=20, stop_idx=1):
        ''' Accepts pre-processed image tensor (inputs) and returns predicted sentence 
        (list of tensor ids of length max_len) '''
        
        res = []
        
        #states = None
        batch_size = inputs.shape[0]
        hidden = self.init_hidden(batch_size)
        
        #for i in range(max_len):
        while True:
            # lstm_out, states = self.lstm(inputs, states)         # hiddens: (1, 1, hidden_size)
            lstm_out, hidden = self.lstm(inputs, hidden)         # hiddens: (1, 1, hidden_size)

            # outputs = self.hidden2out(lstm_out.squeeze(1))       # outputs: (1, vocab_size)
            outputs = self.linear(lstm_out)  # outputs shape : (1, 1, vocab_size)
            outputs = outputs.squeeze(1) # outputs shape : (1, vocab_size)
            _, pred_ind = torch.max(outputs, dim=1) # predict the most likely next word, max_indice shape : (1)
            res.append(pred_ind.cpu().numpy()[0].item()) # storing the word predicted
                       # Get the predicted word
                        # TODO: Training needs to include the STOP index, otherwise it won't be emitted.
            #if predicted_index == stop_idx: #<end> word
            if pred_ind == 1 :    
                break
            
            inputs = self.word_embeddings(pred_ind)      # inputs: (1, embed_size)
            inputs = inputs.unsqueeze(1)                 # inputs: (1, 1, embed_size)
        return res