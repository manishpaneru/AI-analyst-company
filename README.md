# AI Business Analyst System 🤖📊

## Overview
An intelligent system that leverages multiple AI agents to perform comprehensive business data analysis and generate professional reports. This project represents my first venture into AI agent development, combining my data analysis expertise with advanced AI capabilities.

### Author
**Manish Paneru**  
Data Analyst  
[LinkedIn Profile](https://www.linkedin.com/in/manishpaneru1)

## 🎯 Features

### 1. Automated Analysis Pipeline
- **Data Cleaning & Preprocessing**
  - Handles missing values
  - Removes duplicates
  - Standardizes data formats
  - Creates derived metrics

- **Statistical Analysis**
  - Correlation analysis
  - Group-based analysis
  - Revenue pattern identification
  - Price-volume relationship analysis

- **Data Visualization**
  - Price vs. Quantity scatter plots
  - Revenue distribution charts
  - Market share analysis
  - Interactive visualizations

- **Report Generation**
  - Professional PDF reports
  - Executive summaries
  - Detailed analysis sections
  - Data-driven recommendations

### 2. AI Agent Collaboration
The system employs multiple specialized AI agents:
- Data Cleaning Agent
- Statistical Analysis Agent
- Visualization Agent
- Report Writing Agent

## 🛠️ Technology Stack

### Core Technologies
- Python 3.x
- CrewAI (AI Agent Framework)
- Pandas (Data Analysis)
- NumPy (Numerical Computing)
- Matplotlib & Seaborn (Visualization)
- ReportLab (PDF Generation)

### Additional Libraries
- Cache Utils (Custom caching system)
- File I/O Tools
- Statistical Analysis Tools
- Data Visualization Tools

## 📋 Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## 🚀 Installation

1. Clone the repository
```bash
git clone [repository-url]
cd AI_Analyst_company
```

2. Create and activate virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages
```bash
pip install -r requirements.txt
```

4. Set up environment variables
```bash
# Create .env file with necessary configurations
# See .env.example for required variables
```

## 💻 Usage

1. Prepare your data
```bash
# Place your dataset in the project root directory
# Supported formats: CSV, Excel
```

2. Run the analysis
```bash
python crew.py
```

3. Find outputs in:
- `final_report.pdf` - Comprehensive analysis report
- `visualizations/` - Generated charts and graphs
- `cleaned_dataset.csv` - Processed dataset

## 📁 Project Structure
```
AI_Analyst_company/
├── crew.py              # Main entry point
├── tools.py            # Analysis tools implementation
├── agents.py           # AI agent definitions
├── tasks.py            # Task definitions
├── tools_init.py       # Tool initialization
├── cache_utils.py      # Caching functionality
├── requirements.txt    # Dependencies
├── .env               # Environment configuration
├── dataset.csv        # Input data
├── final_report.pdf   # Generated report
└── visualizations/    # Generated charts
```

## 🔍 Sample Output

The system generates a comprehensive PDF report including:
- Executive Summary
- Methodology
- Data Analysis Results
- Visualizations
- Strategic Recommendations

## 🎓 Learning Journey

This project represents my first exploration into AI agent development. Key learning areas include:
- AI agent architecture and collaboration
- Automated report generation
- Advanced data visualization
- Error handling and robustness
- System integration

## 🤝 Contributing

While this is a personal project, I welcome suggestions and feedback. Feel free to:
- Open issues for bugs or suggestions
- Propose improvements
- Share your experience with similar systems

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- CrewAI community for their excellent framework
- Open source community for various tools and libraries
- My professional network for continuous support and feedback

## 📧 Contact

Manish Paneru  
Data Analyst  
[LinkedIn Profile](https://www.linkedin.com/in/manishpaneru1)

---
*This project is part of my journey into AI development. I'm continuously learning and improving, and I welcome any feedback or suggestions!* 