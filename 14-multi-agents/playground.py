from agno.os import AgentOS
from reasoning_finance_team import finance_agent, reasoning_finance_team, web_agent

agent_os = AgentOS(
    id="reasoning_finance_team",
    description="Reasoning Finance Team",
    agents=[web_agent, finance_agent],
    teams=[reasoning_finance_team],
)

app = agent_os.get_app()

if __name__ == "__main__":
    # Default port is 7777; change with port=...
    agent_os.serve(app="reasoning_finance_team:app", reload=True)