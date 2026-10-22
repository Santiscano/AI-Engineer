# from dotenv import load_dotenv
# import os

from langchain_core.runnables import RunnableLambda

# load_dotenv()


# obtener la variable
# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')


step1 = RunnableLambda(lambda x: f"Number {x}")

def duplicate_text(text):
    return [text] * 2

step2 = RunnableLambda(duplicate_text)

chain = step1 | step2

result = chain.invoke(43)
print(result)