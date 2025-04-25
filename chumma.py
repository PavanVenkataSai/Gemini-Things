from langchain_google_genai import ChatGoogleGenerativeAI
import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
llm=ChatGoogleGenerativeAI(model="gemini-pro")


def patient(infor_mation): 

    prompts='''
  content:{question}
  instruction:
  Identify the  First Name,  Last Name, Date of Birth, Phone number, Email. 
  Provide the information in a dictionary format with each key-value pair separated by a newline. If any information is missing, use "None" as the value.  
  For gender, if both male female is there consider as "None". For preferred contact, respond with "Email", "Phone", or "None".you must give what i asked below format and give in order what i give in above content.

output format:
First Name: First Name
Last Name: Last Name
Date of Birth: Date of Birth
Phone number: Phone number
Email: Email

'''

    prompt = ChatPromptTemplate.from_template(prompts)
    # output_parser = StrOutputParser()
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    s=chain.invoke({'question':infor_mation})

    # print('responce',s)
    lines = s.split('\n')
    # print(lines)
    patient_dict = {}
    for line in lines:
        key,value = line.split(':',1)
        patient_dict[key]=value

    if patient_dict:
        print('------------(passed lead/patient extraction)----------')
    else:
        print('------------(failed lead/patient extraction)----------')

    return patient_dict



details=patient('''Patient Details *Patient First Name: Allen M *Patient Last Name: Kennedy *Date of birth (MM/DD/YYYY): 02/11/1994 *Email: AllenK30@hotmail.com Phone number: +212-456-7890''')
print(details)