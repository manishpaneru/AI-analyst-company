# crew.py
import os
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"


import re
from crewai import Agent, Task, Crew
from dotenv import load_dotenv
from agents import data_engineer, data_analyst, report_writer
from tasks import data_cleaning_task, data_analysis_task, report_generation_task



# Load environment variables
load_dotenv()

def validate_business_question(question: str) -> bool:
    """Validate if business question contains only text and numbers"""
    if not question or not isinstance(question, str):
        return False
    # Check if question contains at least one letter or number
    # and only contains letters, numbers, spaces, and basic punctuation
    pattern = r'^[a-zA-Z0-9\s\.,\?\!\'\"]+$'
    return bool(re.match(pattern, question))

def get_business_questions():
    """Get business questions from user input"""
    print("\n" + "="*50)
    print("📊 BUSINESS QUESTIONS INPUT")
    print("="*50 + "\n")
    
    print("Please enter your business questions (type 'done' when finished)")
    questions = []
    
    question_num = 1
    while True:
        question = input(f"Question {question_num}: ").strip()
        if question.lower() == 'done':
            break
        
        if validate_business_question(question):
            questions.append(question)
            question_num += 1
        else:
            print("Invalid question format. Please use only text and numbers.")
    
    return questions

def run_analysis(agents, tasks, business_questions):
    """Create and run the analytics crew"""
    # Create a crew with the agents and tasks
    analytics_crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True
    )
    
    # Prepare inputs
    inputs = {
        "business_questions": business_questions,
        "data_path": "./dataset.csv"
    }
    
    print("\n" + "="*50)
    print("🔍 ANALYSIS IN PROGRESS")
    print("="*50 + "\n")
    
    # Run the crew
    try:
        result = analytics_crew.kickoff(inputs=inputs)
        return result
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        return None

# Main execution
if __name__ == "__main__":
    # Import all required components
    from agents import data_engineer, data_analyst, report_writer
    from tasks import data_cleaning_task, data_analysis_task, report_generation_task
    
    # Define agents list
    agents = [data_engineer, data_analyst, report_writer]
    
    # Define tasks list
    tasks = [data_cleaning_task, data_analysis_task, report_generation_task]
    
    # Get business questions from user
    business_questions = get_business_questions()
    
    if not business_questions:
        print("No business questions provided. Using default questions.")
        business_questions = [
            "What are the main trends in the data?",
            "What factors have the strongest correlation?",
            "Are there any notable patterns or anomalies?"
        ]
    
    print("\n" + "="*50)
    print("📋 ANALYSIS SUMMARY")
    print("="*50 + "\n")
    print(f"Dataset: ./dataset.csv")
    print(f"Business Questions:")
    for i, q in enumerate(business_questions, 1):
        print(f"{i}. {q}")
    print()
    
    # Run the analysis
    result = run_analysis(agents, tasks, business_questions)
    
    if result:
        print("\n" + "="*50)
        print("✅ ANALYSIS COMPLETE")
        print("="*50 + "\n")
        print(result)
    
    print("\n" + "="*50)
    print("📊 ANALYSIS FINISHED")
    print("="*50 + "\n")