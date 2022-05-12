import torch
from .global_varialbles import *

def indexesFromSentence(lang,sent):
  # tokens = []
  # for word in sent.split():
  #   if lang.word2idx.get(word):
  #     tokens.append(lang.word2idx[word])
  # return tokens
  return [lang.word2idx[word]  if lang.word2idx.get(word) else UNK_token for word in sent.split()  ]

def tensorFromSentence(lang,sent,device=device):
  indexes = indexesFromSentence(lang,sent)
  indexes.append(EOS_token)
  return torch.tensor(indexes,dtype=torch.long,device=device).view(-1,1)

def tensorsFromPair(input_lang,output_lang,pair,device=device):
  input_tensor = tensorFromSentence(input_lang,pair[0],device)
  
  target_tensor = tensorFromSentence(output_lang,pair[1],device)
  return (input_tensor,target_tensor)