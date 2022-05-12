


from ml.global_varialbles import *
import pickle
import torch
from ml.models import AttnLSTMDecoder,LSTMEncoder
from os import path
file_path = path.abspath(__file__) # full path of your script
dir_path = path.dirname(file_path)

class Lang:
  def __init__(self,lang_name):
    self.lang_name = lang_name
    self.idx2word = {0:'SOS',1:'EOS',2:'<unk>'}
    self.word2idx = {}
    self.n_words = 3
    self.word2count = {}
  
  def addSentence(self,sentence):
    for word in sentence.split():
      self.addWord(word)
  def addWord(self,word):
    if word not in self.word2idx:
      self.word2idx[word] = self.n_words
      self.idx2word[self.n_words] = word
      self.word2count[word] = 1
      self.n_words +=1
    else:
      self.word2count[word] +=1

class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
      print(name)
      if module == "__main__":
          module = "ml.read_langs_models"
      return super().find_class(module, name)




def read_langs():
    text_lang_path = path.join(dir_path,'text_lang.pkl')
    summary_lang_path = path.join(dir_path,"summary_lang.pkl")
  

    with open(text_lang_path,'rb') as file:
    
      unpickler = MyCustomUnpickler(file)
      input_lang = unpickler.load()

    with open(summary_lang_path,'rb') as file:
    
      unpickler = MyCustomUnpickler(file)
      output_lang = unpickler.load()
    
      
    


  
    return input_lang,output_lang

def read_models(input_lang,output_lang,device):
    encoder_path = path.join(dir_path,'summary_encoder2.pth')
    decoder_path = path.join(dir_path,'summary_decoder2.pth')
    
    
    enc_vocab_size = input_lang.n_words
    dec_vocab_size = output_lang.n_words
    hidden_size = 256
    encoder = LSTMEncoder(vocab_size=enc_vocab_size,hidden_size=hidden_size).to(device)
    decoder = AttnLSTMDecoder(hidden_size=hidden_size,output_size=dec_vocab_size).to(device)

    encoder.load_state_dict(torch.load(encoder_path,map_location=device))
    decoder.load_state_dict(torch.load(decoder_path,map_location=device))

    decoder.eval()
    encoder.eval()
    return encoder,decoder




input_lang,output_lang = read_langs()
encoder,decoder = read_models(input_lang=input_lang,output_lang=output_lang,device=device)




    
    

