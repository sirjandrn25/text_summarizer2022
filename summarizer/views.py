
from base64 import encode
import imp
from django.shortcuts import redirect, render
from django.views import View
from ml.read_langs_models import input_lang,output_lang,encoder,decoder
from ml.global_varialbles import *
from ml.evaluate_model import seq2seqTest,seq2seqEvaluate
from ml.data_preprocessing import text_preprocessor




# Create your views here.



class HomeView(View):
    def get(self,request):
        
        context = {
            "measure_way":"evaluate"
        }
        return render(request,"home/index.html",context)
    
    def post(self,request):
        input_text = request.POST['original_text']
        clean_input_text = text_preprocessor(input_text)
        text_length = len(clean_input_text.split())
        max_summary_length = MAX_LENGTH
        model_measure = request.POST['model_measure']
        
        if model_measure == "evaluate":
            original_summary = request.POST['original_summary']
            clean_original_summary = text_preprocessor(original_summary)
            pred_summary_idx,loss,time_taken = seq2seqEvaluate(encoder=encoder,decoder=decoder,sentence=clean_input_text,summary=clean_original_summary,output_lang=output_lang,input_lang=input_lang,device=device,max_length=text_length)
            accuracy = 100-loss
            measure_way = "evaluate"
            


        else:
            summary_length = request.POST['summary_length']
            # print(type(summary_length))
            max_summary_length = int(summary_length)

            pred_summary_idx,time_taken = seq2seqTest(encoder=encoder,decoder=decoder,sentence=clean_input_text,lang=input_lang,device=device,max_length=text_length,max_summary_len=max_summary_length)
            
            
            accuracy = None
            original_summary = ''
            measure_way = "test"
            
            

        pred_summary = " ".join([output_lang.idx2word[idx] for idx in pred_summary_idx])
        context = {
            'max_summary_len':max_summary_length,
            'pred_summary':pred_summary,
            "time_taken":time_taken,
            "accuracy":accuracy,
            "original_summary":original_summary,
            "input_text":input_text,
            "measure_way":measure_way
            
        }
        
        return render(request,"home/index.html",context)

