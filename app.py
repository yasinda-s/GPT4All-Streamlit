import streamlit as st

#import dependencies
from langchain.llms import GPT4All #you can also use OpenAI, HuggingFace Hub, or GPT4All
from langchain import PromptTemplate, LLMChain

#path to weights
PATH = 'D:/gpt4all/weights/ggml-model-gpt4all-falcon-q4_0.bin' #using the falcon q4 model
llm = GPT4All(model=PATH, verbose=True) #load model

#prompt template
prompt = PromptTemplate(input_variables=['question'], 
                        template="""
                        Question: {question}
                        Answer: Let me give a short answer.
                        """) #Answer is how the answer from the model will begin

#set up LLM chain
chain = LLMChain(prompt=prompt, llm=llm) #baseline LLM chain set up

st.title("Langchain GPT4All")

prompt_from_user = st.text_input("Add any prompt here!")

if prompt_from_user:
    #pass the user prompt to LLM chain
    response = chain.run(prompt_from_user)
    #return the response
    st.write(response)

