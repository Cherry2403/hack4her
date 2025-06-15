import os
from tapeagents.llms import OpenrouterLLM
from dotenv import load_dotenv
load_dotenv()

#LLM
open_router_api_key = os.environ["OPEN_ROUTER_API_KEY"]
llm = OpenrouterLLM(
    model_name="meta-llama/llama-3.3-70b-instruct:free",  # https://openrouter.ai/meta-llama/llama-3.3-70b-instruct:free
    api_token=open_router_api_key,
    parameters={"temperature": 0.1},
)

#Environment setting
from tapeagents.environment import ToolCollectionEnvironment
from tapeagents.tools.calculator import Calculator
from tapeagents.tools.simple_browser import SimpleBrowser
from tapeagents.tools.web_search import WebSearch
from tapeagents.environment import ToolEnvironment

# key for the web search api, you can get a free key at https://serper.dev/
search_api_key = os.environ["SERPER_API_KEY"]
web_search_tool = WebSearch(api_key=search_api_key)

query = "most recent Fiscal Equity Bulletin upper quartile of median municipal earnings"
results = web_search_tool(query)

print("WebSearch Results:")
print(results)

def check_income_threshold(application):
    """Check if household income is below the required threshold."""
    # Implement your logic here
    return 

def check_employment_status(application):
    """Check if at least one guardian is employed or in education."""
    return 

eligibility_env = ToolEnvironment([
    check_income_threshold,
    check_employment_status,
    # Add more checks as needed
])

# simple agent

# from tapeagents.agent import Agent
# from tapeagents.nodes import StandardNode, Stop

# system_prompt = "You are an eligibility checking agent for childcare subsidy applications."

# eligibility_agent = Agent.create(
#     llms=llm,
#     environment=eligibility_env,
#     nodes=[
#         StandardNode(
#             name="check_eligibility",
#             system_prompt=system_prompt,
#             guidance="Check all eligibility criteria using the available tools.",
#             use_known_actions=True,
#         ),
#         Stop(),
#     ],
#     known_actions=eligibility_env.actions,
#     tools_description=eligibility_env.tools_description
# )
