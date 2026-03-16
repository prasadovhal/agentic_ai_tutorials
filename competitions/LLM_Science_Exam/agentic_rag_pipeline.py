import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constant import huggingface_api_key

os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key

##########################################################################

import pandas as pd

DATA_PATH = "D:/Study/Git_repo/agentic_ai_tutorials/competitions/LLM_Science_Exam/data/"
train = pd.read_csv(DATA_PATH + "train.csv")
test = pd.read_csv(DATA_PATH + "test.csv")

#train.head()

train.drop(['id'],axis=1,inplace=True)
test.drop(['id'],axis=1,inplace=True)

