import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constant import huggingface_api_key, GOOGLE_API_KEY, openai_key
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["OPENAI_API_KEY"] = openai_key

###############################################################

# Concept

from openai import OpenAI
client = OpenAI()

def weather_tool(city):
    return f"It is raining in {city}"

query = "Should I carry umbrella in Pune?"

thought = "Need weather info"

observation = weather_tool("Pune")

final_answer = f"{observation}. Yes carry umbrella."

print(final_answer)

###############################################################

# Tools
"""
Agents become powerful when they can use tools.
"""

def calculator(expression):
    return eval(expression)

tools = {
    "calculator": calculator
}

query = "What is 45 * 78?"

tool_name = "calculator"
tool_input = "45*78"

result = tools[tool_name](tool_input)

print(result)

###############################################################

# Tool Calling Agents

tools = [
    {
        "name": "calculator",
        "description": "evaluate math expression",
        "parameters": {
            "expression": "string"
            }
    }
]

from langchain.tools import tool

@tool
def calculator(expression:str):
    """
    

    Args:
        expression (str): _description_

    Returns:
        _type_: _description_
    """
    return eval(expression)

tools = [calculator]