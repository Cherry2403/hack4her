import os
import json
from tapeagents.llms import OpenrouterLLM
from dotenv import load_dotenv
load_dotenv()
from func_util import income_eligible
from func_util import ai_employment_eligible
from func_util import get_subsidy_coverage
from func_util import run_poverty_agent
from func_util import run_decision_agent

#LLM
open_router_api_key = os.environ["OPEN_ROUTER_API_KEY"]
llm = OpenrouterLLM(
    model_name="meta-llama/llama-3.3-70b-instruct:free",  # https://openrouter.ai/meta-llama/llama-3.3-70b-instruct:free
    api_token=open_router_api_key,
    parameters={"temperature": 0.1},
)

#Environment setting
from tapeagents.environment import ToolCollectionEnvironment
from tapeagents.tools.web_search import WebSearch
from tapeagents.tools.calculator import Calculator
from tapeagents.tools.simple_browser import SimpleBrowser

# key for the web search api, you can get a free key at https://serper.dev/

environment = ToolCollectionEnvironment(tools=[WebSearch(), SimpleBrowser(), Calculator()])
environment.initialize()

# Icoming research agent

from tapeagents.nodes import StandardNode, Stop
from tapeagents.agent import Agent

system_prompt = "You are an assistant that can use web search to answer questions."

search_agent = Agent.create(
    llms=llm,
    nodes=[
        StandardNode(
            name="think",
            system_prompt=system_prompt,
            guidance="Let's think about how to use available tools to answer the user's question. Do not answer the question yet.",
        ),
        StandardNode(
            name="act",
            system_prompt=system_prompt,
            guidance="Call the proposed tool.",
            use_known_actions=True,
            use_function_calls=True,
        ),
        StandardNode(
            name="answer",
            system_prompt=system_prompt,
            guidance="Give a final answer to the user's question based on the tool result.",
        ),
        Stop(),
    ],
    known_actions=environment.actions(),
    tools_description=environment.tools_description()
)

# --- Poverty Level Agent ---

poverty_system_prompt = (
    "You are an assistant that assesses the poverty level of a household for subsidy eligibility. "
    "Consider the following fields: household_income, housing_situation, partner_employed, num_children. "
    "Output only one word: low, medium, or high."
)

poverty_agent = Agent.create(
    llms=llm,
    nodes=[
        StandardNode(
            name="think",
            system_prompt=poverty_system_prompt,
            guidance="Think about how to assess the poverty level using the provided data. Do not answer yet.",
        ),
        StandardNode(
            name="answer",
            system_prompt=poverty_system_prompt,
            guidance="Based on the data, output only the poverty level: low, medium, or high.",
        ),
        Stop(),
    ],
    known_actions=environment.actions(),
    tools_description=environment.tools_description()
)

#DECISION AGENT
decision_system_prompt = (
    "You are an assistant that makes final decisions on subsidy applications. "
    "Given the poverty level, child ages, partner employment, and application flags, "
    "If sensitive flag exists 'Specific human view: yes', the percentage of Based on the above, find the rate this application result in form of 'Accepted rate: x% \n Decline rate: y% \n Specific human view: yes/no'"
)

decision_agent = Agent.create(
    llms=llm,
    nodes=[
        StandardNode(
            name="think",
            system_prompt=decision_system_prompt,
            guidance="Think about the decision based on the provided data. Do not answer yet.",
        ),
        StandardNode(
            name="answer",
            system_prompt=decision_system_prompt,
            guidance="'Accepted rate: x% \n Decline rate: y% \n Specific human view: yes/no'",
        ),
        Stop(),
    ],
    known_actions=environment.actions(),
    tools_description=environment.tools_description()
)

#USE AGENT

from tapeagents.dialog_tape import DialogTape, UserStep
from tapeagents.orchestrator import main_loop

user_question = "Using the most recent CBS data for Dutch municipalities (2023), estimate the 75th percentile (upper quartile) of standardized disposable household income in Amsterdam, given that the median income is approximately €30,100 and national income distribution suggests the 75th percentile is about 45% higher than the median. Give me only nothing other than number of result in euros"
tape = DialogTape(steps=[UserStep(content=user_question)])
final_tape1 = None

for event in main_loop(search_agent, tape, environment):
    if event.agent_event and event.agent_event.step:
        step = event.agent_event.step
        print(f"Agent step ({step.metadata.node}):\n{step.llm_view()}\n---")
    elif event.agent_tape:
        tape = event.agent_tape

agent_answer = tape[-2].reasoning 
upper_quartile = int(agent_answer.replace(",", ""))

print("Agent1 earnings threshold report:", agent_answer) 

#application input
with open("data/application.json", "r") as f:
    data = json.load(f)
app_id = input("Enter the application ID to check (e.g., A001): ").strip()
application = next((app for app in data if app.get("application_id") == app_id), None)

# CHECK ELIGIBLE CONDITION
if application:
    if income_eligible(agent_answer, application):
        print("Eligible: Household income is below the upper quartile.")
        result = ai_employment_eligible(application)
        print("AI_check_employment_eligible:", result)
        print(
            f"Please recheck:\n"
            f"  Applicant employment status: {application.get('employment_status', '')}\n"
            f"  Partner employed: {application.get('partner_employed', False)}"
        )

        coverage = get_subsidy_coverage(application, upper_quartile)
        print(f"Max subsidy coverage: {coverage}%")

        #POVERTY AGENT
        agent_answer2 = run_poverty_agent(poverty_agent, application, environment, verbose=True)
        print("Agent2 poverty report:", agent_answer2)

        #DECISION AGENT
        agent_answer3 = run_decision_agent(decision_agent, application, agent_answer2, environment, verbose=True)
        print("Agent3 decision report:", agent_answer3)

    else:
        print("Not eligible: Household income exceeds the upper quartile.")
else:
    print("Application ID not found.")