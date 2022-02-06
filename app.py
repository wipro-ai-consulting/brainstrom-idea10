from transformers import pipeline

import streamlit as st

from transformers import AutoTokenizer
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mrm8488/GPT-2-finetuned-covid-bio-medrxiv")

model = AutoModelForCausalLM.from_pretrained("mrm8488/GPT-2-finetuned-covid-bio-medrxiv")


 
st.title('The opportunity to strengthen GSK ecosystem of Healthcare profressionals during COVID19 lockdowns')

st.image('https://online.stanford.edu/sites/default/files/styles/embedded_large/public/2020-08/artificial-intelligence-in-healthcare-MAIN.jpg?itok=uRNflQFw')


st.header('Hi HCP, Discover an interactive e-learning powered by AI.  And meet new peers on social media using AI, and quizz them !')

st.image('https://www.singlegrain.com/wp-content/uploads/2021/04/SG-7-Ways-to-Leverage-AI-in-Social-Media-Marketing.jpg', width=50)


st.header('Hi HCP, i am a AI trained on a resource of over 500,000 scholarly articles on COVID19')
st.header('Start typing something about COVID, and i will complete the sentance for you. !')

          
prompt = "Vaccinated patients on ICU"

if st.button('Ask the AI to complete the sentance'):
 inputs = tokenizer( prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

 prompt_length = len(tokenizer.decode(inputs[0]))
 outputs = model.generate(inputs, max_length=100, do_sample=True, top_p=0.95, top_k=60)
 generated = prompt + tokenizer.decode(outputs[0])[prompt_length + 1 :]
 st.header(generated)


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


st.subheader('Learning about GSK products can be fun for Healthcare professionals ')

st.header('The early stage concept demo is to enable brainstorming with the customer. Enjoy co-innovating and be a leader. ')

st.subheader('Created by Wipro')

st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Wipro_Primary_Logo_Color_RGB.svg/240px-Wipro_Primary_Logo_Color_RGB.svg.png')



