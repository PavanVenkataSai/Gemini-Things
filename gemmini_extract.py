from langchain_google_genai import ChatGoogleGenerativeAI
import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

llm=ChatGoogleGenerativeAI(model="gemini-pro")


prompts='''
Content:{question}
Instruction:
    Identify the Following attributes:
        Patient First Name, Patient Last Name, Date of birth (YYYY-MM-DD), Email and Phone Number. If any piece of information is not provided for a perticular attribute then respond with "None". 
    Provide the information in a dictionary format with each key-value pair separated by a newline. If any information is missing, use "None" as the value. 
    you must give what i asked and give in order what i give in above content.
    output_format:
        Patient First Name: Patient First Name
        Patient Last Name: Patient Last Name
        Date of birth (YYYY-MM-DD): Date of birth (YYYY-MM-DD)
        Email: Email
        Phone Number: Phone Number

'''

prompt = ChatPromptTemplate.from_template(prompts)
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# s=chain.invoke({'question':"First name Vasanth Preferred longuage Middle initial N Lost nome Shankar Home phone 995246 Date of Birth 07/12/2000 Mole Female Work Phone 995246 Parent/guardian first and lost name (if applicable) Relationship Email Varanth@gmail.com Email Address Earth 744 Notarajan A Dad Cell 9954635791 Preferred contact Email Phone OK to leave message Parent/guardian 2 first and last name of applicable) Relationship Email Vatchala B - Mom Cell 92371671 Preferred contact Email Phone OK to leave message"})
#s=chain.invoke({'question':"First name Walter Middle initial K Last name White Date of Birth 09/09/2001 Male Female Home phone 9573512458 Work Phone 9573558258 Email walter@gmail.com Address Madest,Nove,Alasha,USA Parent/guardian first and last name white smith Relationship father Email smith@gmail.com Cell 8523697410 Preferred contact Email Phone OK to leave message Parent/guardian 2 first and last name Sheree Zampino Relationship mother Email shereezampino@gmail.com Cell 9874563210 Preferred contact Email Phone OK to leave message"})
# s=chain.invoke({'question':"Preferred language First name Arthi Middle initial A Last name Aramugam Date of Birth 21/05/2001 Male Female Home phone 9965351638 Work Phone Email Address Coimbatate Parent/guardian first and lost name (if applicable) Relationship Email Cell 9952551643 Preferred contact Email Phone OK to leave message Parent/guardian 2 first and last name (if applicable) Relationship Email Cell Preferred contact Email Phone OK to leave message"})
# s=chain.invoke({'question':" Preferred language First name KEERTHANA Middle initial Last name DEVARAJ Date of Birth 103/ 2002 Male Female Home phone 6381259354 Work Phone Email keeltharo@gmail.com 1/175 METTUPALAYAM COINSTURE Parent/guardian first and last name (if applicable) Relationship Email Cell Preferred contact Email Phone OK to leave message Parent/guardian 2 first and last name (if applicable) Relationship Email Cell Preferred contact Email Phone OK to leave message"})
# s=chain.invoke({'question':"Patient information Preferred language First name Mohanapiuya Middle initial Last name R Home phone : Date of Birth/7/12/2000 Male Famile Work Phone. 9271120310 Parent/guardian first and lost nome (if applicable) Email mohanapalynis45@yahoo.com 343/199,Bhivashatri colory Relationship Email com Cell 9271120310 Preferred contact Email Phone OK to leave message Parent/guardian 2 first and lost name (if applicable) Relationship Email Cell Preferred contact Email Phone OK to leave message"})
#s=chain.invoke({'question':"Preferred language First name Carls Middle initial Last name Holland Date of Birth09/08/2003 Male Female Home phone (901) 989 9898 Work Phone (901) 979 6853 Emailcarls@yahoo.com AddressMaple St., Nome, Alaska, USA Parent/guardian first and last name (if applicable) Relationship Emailnagarajan12@gmail.com Cell9548762132 Nagarajan swamy father Preferred contact Email Phone OK to leave message Parent/guardian 2 first and last name (if applicable) Relationship Emailpriya34@gmail.com Cell9456873151 priya selvi mother Preferred contact Email Phone OK to leave message"})
# s=chain.invoke({'question':'''Hi, I want to update the details of this particular id, "Id": "a00Hu000015hOH2IAM",
#                                  Details to be updated
#                                  "Name": "P002",
#                                  "email__c": "tonym@gmail.com",
#                                  "phone_no__c": "9229229221",
#                                  "enrollment_start_date__c": "2023-12-22"
#     '''})

s=chain.invoke({'question':'''Patient Details *Patient First Name: Allen M *Patient Last Name: Kennedy *Date of birth (MM/DD/YYYY): 02/11/1994 *Email: AllenK30@hotmail.com Phone number: +212-456-7890'''})



cleaned_content = s.replace('*', '')

print(cleaned_content)
print(type(cleaned_content))

