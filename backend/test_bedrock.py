from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="ap-south-1"
)

response = llm.invoke([
    HumanMessage(content="Explain EBITDA in simple words.")
])

print(response.content)