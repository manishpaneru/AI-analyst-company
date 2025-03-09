# tasks.py
from crewai import Task
from agents import data_engineer, data_analyst, report_writer
from tools_init import (
    data_cleaning_tool,
    statistical_analysis_tool,
    data_visualization_tool,
    pdf_report_tool,
    file_io_tool
)

# Define tasks
data_cleaning_task = Task(
    description="""
    Clean and prepare the dataset at './dataset.csv' for analysis.
    
    Follow these steps in order:
    1. Check if cleaned dataset already exists at './cleaned_dataset.csv':
       - If it exists and is valid, skip cleaning
       - Only proceed with cleaning if necessary
    
    2. If cleaning is needed:
       a) Check if dataset exists and can be read:
          - Use file_io_tool to check file existence
          - Try reading the dataset
          - If file doesn't exist or can't be read, exit with error
    
       b) Analyze dataset structure:
          - Check column names and types
          - Identify missing values
          - Look for duplicates
          - Detect potential outliers
    
       c) Clean the data:
          - Remove duplicate records
          - Handle missing values appropriately
          - Convert data types correctly
          - Handle outliers if necessary
    
       d) Validate cleaned data:
          - Ensure no missing values remain
          - Verify data types are correct
          - Check for any remaining anomalies
    
       e) Save cleaned dataset:
          - Save to './cleaned_dataset.csv'
          - Use appropriate data types
          - Include index=False in save operation
    
    3. Generate cleaning report:
       - Dataset statistics
       - Any changes made (if cleaning was performed)
       - Final dataset structure
    
    Important:
    - Only clean if necessary
    - Do NOT create sample data
    - Document all changes
    - Handle errors gracefully
    
    Provide a summary of the dataset structure and any cleaning performed.
    """,
    agent=data_engineer,
    tools=[data_cleaning_tool, file_io_tool],
    expected_output="A cleaned dataset (if needed) and a dataset report"
)

data_analysis_task = Task(
    description="""
    Analyze the cleaned dataset at './cleaned_dataset.csv' to answer the business questions: {business_questions}
    
    Follow these steps in order:
    1. Read and validate the cleaned dataset:
       - Use file_io_tool to read the dataset
       - Verify data structure and contents
       - Calculate derived metrics (e.g., Revenue = Quantity * UnitPrice)
    
    2. Perform descriptive analysis:
       - Basic statistics (mean, median, std, etc.)
       - Correlation analysis between key variables
       - Group analysis by relevant categories
       - Distribution analysis of key metrics
    
    3. Create visualizations (with error handling):
       Try to create each visualization:
       a) Scatter plot: Price vs Quantity
          - If fails, document the intended visualization
       b) Bar chart: Revenue by Category
          - If fails, provide summary statistics instead
       c) Line chart: Trends over time
          - If fails, describe the temporal patterns
       d) Distribution plots
          - If fails, provide numerical summaries
       
       For each successful visualization:
       - Use clear titles and labels
       - Include legends where needed
       - Save to './visualizations' directory
       - Document any failures or limitations
    
    4. Generate insights:
       - Key findings from statistical analysis
       - Important patterns identified
       - Business implications
       - Data limitations and caveats
    
    Important:
    - Handle missing data appropriately
    - Document any visualization failures
    - Provide alternative analyses when visualizations fail
    - Focus on actionable insights
    
    Expected Output:
    - Statistical analysis results
    - Available visualizations (or documentation of failures)
    - Clear business insights
    """,
    agent=data_analyst,
    tools=[statistical_analysis_tool, data_visualization_tool, file_io_tool],
    expected_output="Analysis results with available visualizations and insights"
)

report_generation_task = Task(
    description="""
    Create a comprehensive, professional PDF report (minimum 1500 words) that presents the analysis findings.
    
    The report must include:
    
    1. Executive Summary highlighting key insights (250-300 words)
       - Overview of main findings
       - Key metrics and trends
       - Critical business implications
       - Primary recommendations
    
    2. Introduction explaining context and objectives (200-250 words)
       - Business context
       - Analysis objectives
       - Dataset overview
       - Analytical approach
    
    3. Methodology section detailing the analytical approach (200-250 words)
       - Data preparation steps
       - Statistical methods used
       - Visualization techniques
       - Analysis limitations
    
    4. Detailed Price-Volume Correlation Analysis (250-300 words)
       - Statistical correlation findings
       - Price elasticity patterns
       - Volume trends by price range
       - Notable exceptions and special cases
    
    5. Revenue Analysis by Price Range (250-300 words)
       - Revenue distribution across price segments
       - Revenue vs. volume efficiency
       - High-performing price ranges
       - Revenue stability analysis
    
    6. Strategic Insights and Business Implications (200-250 words)
       - Optimal price positioning
       - Market segmentation opportunities
       - Competitive positioning
       - Growth opportunities
    
    7. Actionable Recommendations (200-250 words)
       - Price optimization strategy
       - Inventory and product mix management
       - Marketing and promotion strategies
       - Long-term strategic initiatives
    
    8. Limitations and Further Research (150-200 words)
       - Data limitations
       - Analytical constraints
       - Areas for further investigation
       - Suggested improvements
    
    For each visualization:
    - Include detailed explanations of patterns and trends
    - Highlight key insights in callout boxes
    - Connect findings to business implications
    - Provide context for interpretation
    
    Important Requirements:
    - Professional formatting with consistent styling
    - Clear section headings and subheadings
    - Page numbers and table of contents
    - Executive-friendly language and presentation
    - Actionable insights and recommendations
    - Proper citation of data sources
    - High-quality visualizations with explanations
    - Logical flow between sections
    
    Save the report as './final_report.pdf'
    """,
    agent=report_writer,
    tools=[pdf_report_tool, file_io_tool],
    expected_output="A comprehensive, professional PDF report exceeding 1500 words with detailed analysis, visualizations, and actionable recommendations"
)

# Set task dependencies
data_analysis_task.context = [data_cleaning_task]
report_generation_task.context = [data_analysis_task]