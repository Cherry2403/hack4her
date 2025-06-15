import json

# Load the JSON dataset
with open("data/application.json", "r") as f:
    data = json.load(f)

def format_application(app):
    child_ages = ", ".join(str(age) for age in app.get("child_ages", []))
    municipal_support = ", ".join(app.get("recent_municipal_support", []))
    flags = app.get("flags", {})
    flags_str = ", ".join(f"{k}: {v}" for k, v in flags.items())
    return (
        f"Application ID: {app.get('application_id', '')}\n"
        f"Applicant Name: {app.get('applicant_name', '')}\n"
        f"Household Income: {app.get('household_income', '')}\n"
        f"Employment Status: {app.get('employment_status', '')}\n"
        f"Number of Children: {app.get('num_children', '')}\n"
        f"Child Ages: {child_ages}\n"
        f"Childcare Hours Requested: {app.get('childcare_hours_requested', '')}\n"
        f"Housing Situation: {app.get('housing_situation', '')}\n"
        f"Partner Employed: {app.get('partner_employed', '')}\n"
        f"Recent Municipal Support: {municipal_support}\n"
        f"Flags: {flags_str}\n"
    )
class SubsidyEnvironment:
    def _init_(self, application_record):
        self.application = application_record

    def actions(self):
        return [
            "check_residency", 
            "check_employment", 
            "check_income", 
            "check_documents"
        ]

    def tools_description(self):
        return {
            "check_residency": "Check if housing_situation suggests Dutch residency.",
            "check_employment": "Check if employment_status is employed/studying or partner_employed is True.",
            "check_income": "Check if household_income is below 70000.",
            "check_documents": "Check if flags['incomplete_docs'] is False."
        }

    def run_action(self, action):
        app = self.application
        if action == "check_residency":
            # Example logic, can be improved if you have a 'residency' field
            return "PASS" if "Netherlands" in app.get("housing_situation", "") or "housing" in app.get("housing_situation", "") else "FAIL"
        if action == "check_employment":
            # If either applicant or partner is employed/studying
            if app.get("employment_status", "").lower() in ["employed", "studying"]:
                return "PASS"
            if app.get("partner_employed", False):
                return "PASS"
            return "FAIL"
        if action == "check_income":
            return "PASS" if app.get("household_income", 0) < 70000 else "FAIL"
        if action == "check_documents":
            flags = app.get("flags", {})
            return "PASS" if not flags.get("incomplete_docs", False) else "FAIL"
        return "UNKNOWN"
    
from tapeagents.dialog_tape import DialogTape, UserStep
from tapeagents.agent import Agent
from tapeagents.nodes import StandardNode
from tapeagents.core import Prompt, FinalStep
from tapeagents.llms import LiteLLM
from tapeagents.orchestrator import main_loop

llm = LiteLLM(model_name="gpt-4o-mini")
system_prompt = "You are an expert assistant evaluating Dutch childcare subsidy applications. Use CSAR 2025 criteria for all steps."

react_agent = Agent.create(
    llms=llm,
    nodes=[
        StandardNode(
            name="plan",
            system_prompt=system_prompt,
            guidance="Write a concise step-by-step plan to assess the subsidy application. Do not answer yet.",
            next_node="reflect"
        ),
        StandardNode(
            name="reflect",
            system_prompt=system_prompt,
            guidance="Evaluate progress and decide next action: check_residency, check_employment, check_income, check_documents. If finished, use FinalStep for the final decision.",
            next_node="act"
        ),
        StandardNode(
            name="act",
            system_prompt=system_prompt,
            guidance="Call the chosen action/tool. Then return to reflect. If all steps are complete, call FinalStep.",
            use_known_actions=True,
            use_function_calls=True,
            steps=FinalStep,
            next_node="reflect"
        ),
    ],
    known_actions=SubsidyEnvironment.actions,
    tools_description=SubsidyEnvironment.tools_description
)

import json
with open("data/application.json", "r") as f:
    data = json.load(f)

for application in data[:1]:  # Or data for all
    env = SubsidyEnvironment(application)
    prompt = (
        f"Evaluate this subsidy application:\n{format_application(application)}\n"
        "Is the application eligible under CSAR 2025? Justify the answer."
    )
    tape = DialogTape(steps=[UserStep(content=prompt)])

    for event in main_loop(react_agent, tape, env):
        if event.agent_event and event.agent_event.step:
            step = event.agent_event.step
            print(f"Agent step ({step.metadata.node}):\n{step.llm_view()}\n---")
        elif event.observation:
            print(f"Tool call result:\n{event.observation.short_view()}\n---")
        elif event.agent_tape:
            tape = event.agent_tape

    print("\n\nAgent final answer:", getattr(tape[-1], 'reason', 'No final reason found.'))