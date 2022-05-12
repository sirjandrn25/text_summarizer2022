import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMEncoder(nn.Module):
  def __init__(self,vocab_size,hidden_size,num_layers=1):
    super(LSTMEncoder,self).__init__()
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    
    self.embeddings = nn.Embedding(num_embeddings=vocab_size,embedding_dim=hidden_size)
    
    self.lstm = nn.LSTM(hidden_size,hidden_size,num_layers=num_layers)
  
  def forward(self,X,hidden):
    
    input = X.view(1,-1)
  

    embedded = self.embeddings(input)
    
    
    output,hidden = self.lstm(embedded,hidden)
    return output,hidden
  
  def init_lstm_state(self,device):
    return (
        torch.zeros(size=(1,1,self.hidden_size),dtype=torch.float32,device=device),
        torch.zeros(size=(1,1,self.hidden_size),dtype=torch.float32,device=device)
    )



class AttnLSTMDecoder(nn.Module):
  def __init__(self,hidden_size,output_size,dropout=0.5):
    super(AttnLSTMDecoder,self).__init__()
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.dropout = dropout

    # calculating queries --> queries = W_q*Q 
    self.fc_hidden = nn.Linear(in_features=hidden_size,out_features=hidden_size,bias=False)

    #calculating keys --> keys = W_k* k
    self.fc_encoder = nn.Linear(in_features=hidden_size,out_features=hidden_size,bias=False)

    # calculating weight paramters 
    self.weight = nn.Parameter(torch.FloatTensor(1,hidden_size))

    

    self.embedding = nn.Embedding(num_embeddings=output_size,embedding_dim=hidden_size)
    self.attn_combine = nn.Linear(in_features=hidden_size*2,out_features=self.hidden_size)
    self.lstm_layer = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size)
    self.fc_classifier = nn.Linear(in_features=hidden_size,out_features=output_size)
    self.dropout = nn.Dropout(dropout)

  
  def forward(self,X,hidden,encoder_outputs):
    encoder_outptus = encoder_outputs.squeeze()
    
    X = X.view(1,-1)
    embedded = self.dropout(self.embedding(X)).view(1,-1)
    
    # calculating alignment scores
    #query is previous hidden_state(if decoder initial state previous hidden state is encoder final hidden state) and keys is encoder outputs
    x = torch.tanh(self.fc_hidden(hidden[0])+self.fc_encoder(encoder_outputs))
    
    alignment_scores = x.bmm(self.weight.unsqueeze(2))
    
    # using softmax to alignment scores to get attention weights
    attn_weights = F.softmax(alignment_scores.view(1,-1),dim=1)
    
    # multiplying the attention weights with encoder outputs to get context vector
    context_vector = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))

    # concatenating context vector with embedded input word
    attn_applied = torch.cat((embedded,context_vector[0]),1)
    decoder_input = self.attn_combine(attn_applied).unsqueeze(0)

    out,hidden = self.lstm_layer(decoder_input,hidden)
    output = F.log_softmax(self.fc_classifier(out[0]),dim=1)
    # print(output.shape)
    return output,hidden,attn_weights