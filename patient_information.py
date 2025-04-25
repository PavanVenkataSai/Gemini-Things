from langchain_google_genai import ChatGoogleGenerativeAI
import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Define your dictionary of attributes and their descriptions

attributes = {
    'Preferred language': "Extract the preferred language from the given text. If not available, provide 'None'.",
    'First Name': "Extract the first name from the given text. If not available, provide 'None'.",
    # 'Middle Initial': "Extract the middle initial from the given text. Middle initial can be an initial like 'John S William' where 'S' is the middle initial. If not available, provide 'None'.",
    'Last Name': "Extract the last name or surname or family name from the given text. Last name can also be an initial like 'John S' where 'S' is the last name. If not available, provide 'None'.",
    'Date of Birth': "Extract the date of birth from the given text. The date format should be in year-month-date format. If not available, provide 'None'.",
    'Gender': "Extract the gender from the given text. If not available, provide 'None'.",
    'Home Phone': "Extract the home phone number from the given text. If not available or if the number has more than 10 digits, provide 'None'.",
    'Work Phone': "Extract the work phone number from the given text. If not available or if the number has more than 10 digits, provide 'None'.",
    'Email': "Extract the email address from the given text. If not available, provide 'None'.",
    'Address': "Extract the address from the given text. If not available, provide 'None'.",
    'Parent/guardian first and last name': "Extract the parent/guardian first and last name from the given text. If not available, provide 'None'.",
    'Relationship': "Extract the relationship from the given text. If not available, provide 'None'.",
    'Email': "Extract the email address from the given text. If not available, provide 'None'.",
    'Cell': "Extract the cell phone number from the given text. If not available or if the number has more than 10 digits, provide 'None'.",
    'Parent/guardian 2 first and last name': "Extract the parent/guardian 2 first and last name from the given text. If not available, provide 'None'.",
    'Relationship': "Extract the relationship from the given text. If not available, provide 'None'.",
    'Email': "Extract the email address from the given text. If not available, provide 'None'.",
    'Cell': "Extract the cell phone number from the given text. If not available or if the number has more than 10 digits, provide 'None'.",
}



# Function to construct and send prompts to the Gemini model
def extract_attributes(message_body):
    # Initialize an empty dictionary to store the extracted attributes
    extracted_attributes = {}

# Iterate over each attribute and its description
    for attribute, description in attributes.items():
        # Initialize a counter for retries
        retries = 0
        attribute_value = None

        while retries < 3:
            # Construct the prompt
            prompt_template = f'''
            {description}
            {message_body}
            If any piece of information is not provided, respond with "None."
            '''
            
            # Create a ChatPromptTemplate from the template
            prompt = ChatPromptTemplate.from_template(prompt_template)
            
            output_parser = StrOutputParser()

            chain = prompt | llm | output_parser

            # Invoke the chain with the constructed prompt
            output = chain.invoke({'question': prompt})
            
            # Extract the attribute value from the output
            attribute_value = output.strip()
            
            # If the attribute value is not 'None', break the loop
            if attribute_value != 'None':
                break
            
            # Increment the retry counter
            retries += 1

        # Add the attribute and its value to the extracted_attributes dictionary
        extracted_attributes[attribute] = attribute_value
    
    return extracted_attributes
    
# Example usage
# message_body = "hey,First name Vasanth Preferred longuage Middle initial N Lost nome Shankar Home phone 995246 Date of Birth 07/12/2000 Male Work Phone 995246 Parent/guardian first and lost name (if applicable) Relationship Email Varanth@gmail.com Email Address Earth 744 Notarajan A Dad Cell 9954635791 Preferred contact Email Phone OK to leave message Parent/guardian 2 first and last name of applicable) Relationship Email Vatchala B - Mom Cell 92371671 Preferred contact Email Phone OK to leave message"
message_body = "hey,First name Vasanth Preferred language Middle initial N Last name Shankar Home phone 995246 Date of Birth 07/12/2000 Male Work Phone 995246 Parent/guardian first and lost name (if applicable) Relationship Email Varanth@gmail.com Email Address Earth 744 Notarajan A Dad Cell 9954635791 Preferred contact Email Phone OK to leave message Parent/guardian 2 first and last name of applicable) Relationship Email Vatchala B - Mom Cell 92371671 Preferred contact Email Phone OK to leave message"
# message_body = "Hey, I am Nayana S Gopal. I want to create a lead account with the same name born on 11th march 2002, addresed in Illinois, Chicago, US - 52781 located in 3rd, lemon street. Contact me on 7199100112 or keerthana@gmail.com."
extracted_attributes = extract_attributes(message_body)
print(extracted_attributes)