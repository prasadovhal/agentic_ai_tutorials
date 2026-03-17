import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chatbot.graph import app

if __name__ == "__main__":

    result = app.invoke({
        "question": "What is LangGraph?"
    })

    print("\nFinal Answer:\n", result["answer"])