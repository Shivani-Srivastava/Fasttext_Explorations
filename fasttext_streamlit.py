'''

Multi-Label Classification

'''


import streamlit as st
import fasttext
import pandas as pd

model = fasttext.train_supervised(input="cooking.train")

model.save_model("model_cooking.bin")

st.title('Multi Classification Fasttext')

#model.predict("Which baking dish is best to bake a banana bread ?")

test_output = model.test("cooking.valid")
df0 = pd.DataFrame(test_output)
df0.index = ['Samples', 'Precision', 'Recall']
st.sidebar.write('Model Metrics')
st.sidebar.dataframe(df0)

#st.sidebar.write('To improve this, select options from below.')

pred_Q = st.text_input('Enter a question appropriate for a Cooking StackExchange:', placeholder = "Why not put knives in the dishwasher?")

num_pred = st.slider('How many predicted labels?',0,51,2)

st.write(pred_Q)

data = model.predict(pred_Q, k=num_pred)

df = pd.DataFrame(data)

#, columns = ['Label', 'Probability']

df=df.transpose()

st.dataframe(df)


