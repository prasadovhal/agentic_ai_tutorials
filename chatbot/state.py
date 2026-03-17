from typing import TypedDict, List

class AgentState(TypedDict):

    question: str
    context: List[str]
    answer: str
    reflection: str