import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from ml.utils import contraction_mapping

stop_words = stopwords.words('english')
def text_preprocessor(text):
  # first maping words from using contracting maping
  text = text.lower() #contvert into lower case
  text = text.split() #make tokens using on the basis of white space

  for i in range(len(text)):
    word = text[i]
    if word in contraction_mapping:
      text[i] = contraction_mapping[word]
  
  text = " ".join(text) #again convert tokens into text

  #remove stopwords from text
  text = text.split()
  temp_text = []
  for word in text:
    if word not in stop_words:
      temp_text.append(word)
  
  text = " ".join(temp_text)
  text = text.replace("'s",'') # convert your's -> your
  text = re.sub(r'\(.*\)','',text) #remove bracket inside words with brackets
  text = re.sub(r'[^a-zA-Z0-9. ]','',text) #remove punctuations
  text  = re.sub(r'\.',' . ',text)
  return text