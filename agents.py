import os 
from dotenv import load_dotenv
from crewai import Agent
from tools_init import (
    data_cleaning_tool,
    statistical_analysis_tool,
    data_visualization_tool,
    pdf_report_tool,
    file_io_tool
)

load_dotenv()

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")


data_engineer = Agent(
    role="Data Engineer",
    goal="Clean and prepare data for analysis",
    backstory="You are an expert in data cleaning and preparation with years of experience handling messy datasets.",
    llm="gemini/gemini-1.5-flash",
    tools=[data_cleaning_tool, file_io_tool],
    verbose=True
)

data_analyst = Agent(
    role="Data Analyst",
    goal="Find meaningful insights and create visualizations",
    backstory="You are a skilled data analyst who can identify patterns and create insightful visualizations.",
    llm="gemini/gemini-1.5-flash",
    tools=[statistical_analysis_tool, data_visualization_tool, file_io_tool],
    verbose=True
)

report_writer = Agent(
    role="Report Writer",
    goal="Create professional PDF reports that communicate insights clearly",
    backstory="You are an experienced report writer who can translate technical findings into clear business reports.",
    llm="gemini/gemini-1.5-flash",
    tools=[pdf_report_tool, file_io_tool],
    verbose=True
)


