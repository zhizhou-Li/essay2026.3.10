from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # 加载 .env 文件
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print(os.getenv("OPENAI_API_KEY") )
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Describe the shape of the Eiffel Tower in one sentence."}
    ]
)

print(response.choices[0].message.content)
