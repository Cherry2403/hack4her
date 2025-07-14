import re
import os
import json
from tapeagents.llms import OpenrouterLLM
from dotenv import load_dotenv
load_dotenv()
from tapeagents.dialog_tape import DialogTape, UserStep
from tapeagents.orchestrator import main_loop

#LLM
open_router_api_key = os.environ["OPEN_ROUTER_API_KEY"]
llm = OpenrouterLLM(
    model_name="meta-llama/llama-3.3-70b-instruct:free", 
    api_token=open_router_api_key,
    parameters={"temperature": 0.1},
)

def income_eligible(agent_answer: str, application: dict) -> bool:
    match = re.search(r"(\d{2,3}(?:[.,]\d{3})*)", agent_answer)
    if not match:
        print("Could not extract upper quartile value from the agent's answer.")
        return False
    upper_quartile = int(match.group(1).replace(",", "").replace(".", ""))
    return application["household_income"] <= upper_quartile

def ai_employment_eligible(application: dict) -> bool:
    """
    Uses an LLM to determine if the applicant or their partner meets the employment/education criteria.
    """
    prompt = (
        f"Applicant employment status: {application.get('employment_status', '')}\n"
        f"Partner employed: {application.get('partner_employed', False)}\n"
        "Does this meet the following eligibility condition?\n"
        "At least one guardian must be engaged in gainful employment, a recognized educational or vocational program, "
        "or a government-authorized reintegration or reskilling trajectory. "
        "Answer only 'yes' or 'no'."
    )
    response = llm.quick_response(prompt)
    response_text = response.text if hasattr(response, "text") else response
    return "yes" in response_text.lower()

def get_subsidy_coverage(application, upper_quartile):
    app_income = int(application["household_income"])
    median_income = 30100
    # Estimate 25th percentile (assuming linear distribution for simplicity)
    percentile_25 = median_income - (upper_quartile - median_income)
    percentile_50 = median_income
    percentile_75 = upper_quartile

    if app_income <= percentile_25:
        return 85
    elif app_income <= percentile_50:
        return 60
    elif app_income <= percentile_75:
        return 30
    else:
        return 
    
def run_poverty_agent(poverty_agent, application, environment, verbose=False):
    """
    Runs the poverty agent on the given application and returns the poverty report.
    If verbose=True, prints each agent step.
    """
    poverty_question = (
        f"Assess the poverty level for this household:\n"
        f"- Household income: {application.get('household_income')}\n"
        f"- Housing situation: {application.get('housing_situation')}\n"
        f"- Partner employed: {application.get('partner_employed')}\n"
        f"- Number of children: {application.get('num_children')}\n"
        "Output only one word: LOW, MEDIUM, or HIGH."
    )
    poverty_tape = DialogTape(steps=[UserStep(content=poverty_question)])

    for event in main_loop(poverty_agent, poverty_tape, environment):
        if verbose and event.agent_event and event.agent_event.step:
            step = event.agent_event.step
            print(f"Poverty Agent step ({step.metadata.node}):\n{step.llm_view()}\n---")
        elif event.agent_tape:
            poverty_tape = event.agent_tape

    # Get the answer from the answer node (usually second-to-last)
    agent_answer2 = poverty_tape[-2].reasoning
    return agent_answer2

def run_decision_agent(decision_agent, application, poverty_level, environment, verbose=False):
    """
    Runs the decision agent on the given application and returns the decision.
    """
    prompt = (
        f"Application data:\n"
        f"- Poverty level: {poverty_level}\n"
        f"- Child ages: {application.get('child_ages')}\n"
        f"- Partner employed: {application.get('partner_employed')}\n"
        f"- Application flags: {application.get('flags')}\n"
        "If sensitive flag exists 'Specific human view: yes', the percentage of Based on the above, find the rate this application result in form of 'Accepted rate: x% \n Decline rate: y% \n Specific human view: yes/no' each stats in new line at and show firstly, on new next line, explain decision shortly in 3-5 sentence"
    )
    decision_tape = DialogTape(steps=[UserStep(content=prompt)])
    for event in main_loop(decision_agent, decision_tape, environment):
        if verbose and event.agent_event and event.agent_event.step:
            step = event.agent_event.step
            print(f"Decision Agent step ({step.metadata.node}):\n{step.llm_view()}\n---")
        elif event.agent_tape:
            decision_tape = event.agent_tape
    
    agent_answer3 = decision_tape[-2].reasoning
    return agent_answer3