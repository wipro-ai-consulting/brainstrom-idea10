from transformers import pipeline

import streamlit as st

from transformers import AutoTokenizer
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mrm8488/GPT-2-finetuned-covid-bio-medrxiv")

model = AutoModelForCausalLM.from_pretrained("mrm8488/GPT-2-finetuned-covid-bio-medrxiv")

prompt = "Today the weather is really nice and I am planning on "
inputs = tokenizer( prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

prompt_length = len(tokenizer.decode(inputs[0]))
outputs = model.generate(inputs, max_length=100, do_sample=True, top_p=0.95, top_k=60)
generated = prompt + tokenizer.decode(outputs[0])[prompt_length + 1 :]

st.write(generated)

 
st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Wipro_Primary_Logo_Color_RGB.svg/240px-Wipro_Primary_Logo_Color_RGB.svg.png')
st.title('Medical Affairs Use case: Strengthen GSK ecosystem of Healthcare profressionals with a fun interactive e-learning , powered by AI')

st.header('Hi, Healthcare professional, Lets fight COVID together.  Experts say there are still learning about coronavirus, given its its fast mutataions')
          



context = st.text_area(label='context' , value ='Extractive Question Answering is the task of extracting an answer from a text given a question')
 
myquestion = st.text_area(label='question', value='What is extractive question answering?')
 


 


          
question_answerer = pipeline("question-answering")


result = question_answerer(question=myquestion, context=context)

st.write(result['answer'])


unmasker2 = pipeline("fill-mask", model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
resultsfillmask = unmasker2('vaccine induce [MASK] response', top_k=4)
st.write(resultsfillmask[0]['sequence'])
st.write(resultsfillmask[1]['sequence'])
st.write(resultsfillmask[2]['sequence'])


