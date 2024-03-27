import os

from dotenv import load_dotenv

load_dotenv()
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

print(os.environ["SERPER_API_KEY"])

# ex?it()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = os.getenv("OPENAI_MODEL_NAME")

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

search_tool = SerperDevTool()


topic="legal proceedings against Donald Trump"
time_window="1 week"
time_cd="w1"


researcher = Agent(
    role="News Researcher",
    goal=f"Identify the most signifiant developments in the news regarding {topic} over the past {time_window}.",
    verbose=True,
    memory=True,
    backstory="Driven by curiosity, you have a flair for research and fact-checking, with the ability to put information in historical, economic, and political context."
    "the world.",
    tools=[search_tool],
    allow_delegation=True
)

writer = Agent(
    role="News Writer",
    goal=f"Produce a succinct yet informative, insightful, and incisive brief on the most significant developments in the news regarding {topic} over the past {time_window}",
    verbose=True,
    memory=True,
    backstory="You produce engaging and insightful narratives while being fair-minded, intellectually honest, and informative. You explain events and their importance and put them in context.",
    tools=[search_tool],
    allow_delegation=False
)

research_task = Task(
    description=f"Identify the most significant developments regarding {topic} over the last {time_window}.",
    expected_output="A detailed brief on the most important developments and their importance as well as URLs for the sources of the information.",
    tools=[search_tool],
    agent=researcher
)

writing_task = Task(
    description=f"Produce a succinct yet informative, insightful, and incisive brief on the most significant developments in the news regarding {topic} over the past {time_window}. Explain the importance of these events and put them in context.",
    # expected_output="An engaging and informative long-form analysis of the developments, between 300 and 500 words (2-4 paragraphs), in markdown format. The synthesis must also explain the events' importance and put them in context. Moreover, the output should be a general syntesis and NOT a list of summaries of each article. It must be written in paragaphs and not as a list.",  # It should end with a sources section that includes URLs for the sources; the URLs must be included ONLY at the end.",
    expected_output=f"An engaging and informative long-form summary and analysis of the most important developments regarding {topic}, between 300 and 500 words, in two paragraphs. The first paragraph should summarize the developments in general, without listing the source articles out individually. The second paragraph should begin with the header 'Why it matters' and put the developments into context and explain their importance. Finally, following the summary, provide a 'References' section with the URLs of the source articles.",
    tools=[search_tool],
    agent=writer,
    async_execution=False,
    output_file=f"{topic}-news-update.md"
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential
)

result = crew.kickoff(inputs={"topic": topic})
print(result)