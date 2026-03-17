from typing import TypedDict, List, Dict

class AgentState(TypedDict):

    question: str
    rewritten_query: str
    sub_questions: List[str]

    contexts: List[str]

    answer: str
    reflection: str

    judge_score: Dict
    metrics: Dict

    chat_history: List[str]