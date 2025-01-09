import streamlit as st
from src.config import model_path, map_to_class
import torch
from transformers import BertForSequenceClassification, BertTokenizer


model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

st.markdown("<h1 style='text-align: center;'>Financial Sentiment Analyzer</h1>", 
            unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>😊Positive😊  || 😐Neutral😐 ||  😓Negative😓</h5>", 
            unsafe_allow_html=True)
st.write("-"*50)
user_inputs = st.text_area(label=' ', placeholder='Input your text here.....')
st.write('\n\n')
button = st.button(label='Submit')

if button :
    if user_inputs is None:
        st.write("Enter something in text area to analyze!")
    if user_inputs is not None:
        inputs = tokenizer(text=user_inputs, max_length=128, padding=True, truncation=True, 
                           return_tensors='pt', add_special_tokens=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred= torch.argmax(logits, dim=-1).item()

        st.write(f"Predicted Sentiment : {map_to_class(pred)}")
