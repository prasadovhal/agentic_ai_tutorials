from graph import app

if __name__ == "__main__":

    result = app.invoke({
        "question": "What is LangGraph?"
    })

    print("\nFinal Answer:\n", result["answer"])