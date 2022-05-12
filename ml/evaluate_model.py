
import time
from ml.covert_tensors import *
from ml.utils import getTime
import torch.nn as nn
import torch
from ml.global_varialbles import *


from ml.read_langs_models import input_lang,output_lang,encoder,decoder


def seq2seqEvaluate(encoder,decoder,sentence,summary,input_lang,output_lang,device,max_length=MAX_LENGTH):

  start_time = time.time()
  criterion = nn.NLLLoss()
  input_tensor = tensorFromSentence(input_lang,sentence,device=device)
  output_tensor = tensorFromSentence(output_lang,summary,device=device)

  encoder_len = len(input_tensor)
  encoder_hidden = encoder.init_lstm_state(device=device)
  encoder_outputs = torch.zeros((max_length+1,encoder.hidden_size),device=device)
  for ei in range(encoder_len):
    encoder_input = input_tensor[ei]
    encoder_output,encoder_hidden = encoder(encoder_input,encoder_hidden)
    encoder_outputs[ei] = encoder_output
  
  decoder_hidden = encoder_hidden
  decoder_input = torch.tensor([[SOS_token]],dtype=torch.long,device=device)

  decoder_outputs = []
  
  loss = 0
  target_length = len(output_tensor)
  decoder_attentions = torch.zeros(MAX_LENGTH+2,MAX_LENGTH+2)
  for di in range(target_length):
    decoder_output,decoder_hidden,decoder_attn = decoder(decoder_input,decoder_hidden,encoder_outputs)
    # decoder_attentions[di] = decoder_attn.data
    loss += criterion(decoder_output,output_tensor[di])
    _,topi = decoder_output.topk(1)
    
    if topi.item() == EOS_token:
      decoder_outputs.append(EOS_token)
      break
    else:
      decoder_outputs.append(topi.item())

    decoder_input = topi.squeeze().detach()
  # print(" ".join([output_lang.idx2word[idx] for idx in decoder_outputs]))
  # return decoder_outputs,loss.item()/len(output_tensor),decoder_attentions[:di+1]
  return decoder_outputs,loss.item()/target_length,getTime(start_time)



def seq2seqTest(encoder,decoder,sentence,lang,max_summary_len,device=device,max_length=MAX_LENGTH):

    start_time = time.time()
    input_tensor = tensorFromSentence(lang,sentence,device=device)

    encoder_length = len(input_tensor)
    encoder_hidden = encoder.init_lstm_state(device=device)
    encoder_outputs = torch.zeros((max_length+1,encoder.hidden_size),device=device)

    for ei in range(encoder_length):
        encoder_input = input_tensor[ei]
        encoder_output,encoder_hidden = encoder(encoder_input,encoder_hidden)
        encoder_outputs[ei] = encoder_output
    
    
    decoder_hidden = encoder_hidden
    decoder_input = torch.tensor([SOS_token],dtype=torch.long,device=device)

    decoder_outputs = []

    for di in range(max_summary_len):
        decoder_output,decoder_hidden,decoder_attn = decoder(decoder_input,decoder_hidden,encoder_outputs)
        _,topi = decoder_output.topk(1)

        decoder_outputs.append(topi.item())

        if topi.item() == EOS_token:
            break
        decoder_input = topi.squeeze().detach()
    
    return decoder_outputs,getTime(start_time)



if __name__ == "__main__":
    
    sentence = "delhi high court reduced compensation awarded motor accident victim 45 found negligence part parties .  compensation 10 lakh earlier awarded victim .  court observed it possible despite vehicle driven permissible limit accident occur jaywalker suddenly appears road ."

    decoder_outputs_idx = seq2seqTest(encoder=encoder,decoder=decoder,sentence=sentence,lang=input_lang)
    print(" ".join([output_lang.idx2word[idx] for idx in decoder_outputs_idx]))
    



            


    