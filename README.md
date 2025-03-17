# AI Interview Assessment Platform

A powerful AI-powered platform that automatically analyzes interview videos and provides comprehensive candidate assessments. This tool leverages Google's Gemini AI to evaluate technical skills, communication abilities, behavioral traits, and overall candidate fit.

## 🚀 Features

- **Video Interview Analysis**: Upload interview recordings for AI-powered assessment
- **Customizable Evaluation**: Adjust importance weights for different assessment categories
- **Comprehensive Results**: Get detailed scores across multiple skills and competencies
- **Interactive Dashboard**: Visualize candidate performance with charts and graphs
- **Detailed Feedback**: Receive timestamps of key moments with specific observations
- **Role-Specific Assessment**: Tailor analysis to specific job roles and experience levels
- **Export Options**: Download assessment data for record-keeping

## 📊 Assessment Categories

The platform evaluates candidates across seven key areas:

1. **Technical Skills** (domain knowledge, problem-solving, coding proficiency)
2. **Communication Skills** (clarity, listening, conciseness, non-verbal cues)
3. **Behavioral & Soft Skills** (leadership, adaptability, emotional intelligence)
4. **Strengths & Weaknesses** (self-awareness, improvement mindset)
5. **Cultural Fit & Attitude** (values alignment, growth mindset, work ethic)
6. **Problem-Solving & Critical Thinking** (logical thinking, creativity)
7. **Decision-Making Ability** (analytical thinking, performance under pressure)

## 📋 Prerequisites

Before you get started, make sure you have:

- Python 3.8 or higher
- A Google Cloud account with access to the Gemini API
- Google API key with Gemini permissions

## 🛠️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Bharath4ru/interview-assessment-ai.git
   cd interview-assessment-ai
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## 📝 Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. Use the interface to:
   - Select the job role and experience level you're assessing
   - Adjust category weights based on position requirements
   - Upload an interview video recording
   - Enter candidate details
   - Run the analysis
   - View comprehensive results and visualizations

## 📊 Result Interpretation

The dashboard provides several views of the candidate assessment:

- **Final Score**: Overall performance rating out of 100
- **Performance by Category**: Radar chart showing strengths and weaknesses
- **Detailed Skills Breakdown**: Bar charts of specific competencies
- **Key Observations**: Timestamped moments from the interview
- **Role Fit Analysis**: Overall recommendation with justification

## 🔧 Customization

You can modify `assessment_categories` in the code to:
- Add or remove categories
- Change category weights
- Adjust subcategories
- Customize evaluation criteria

## 📦 Requirements

The main dependencies are:
- streamlit
- phidata
- google-generativeai
- plotly
- pandas
- python-dotenv

See `requirements.txt` for the complete list.

## 🔒 Privacy & Data Usage

- Interview videos are processed by Google's Gemini AI
- Videos are not permanently stored but are temporarily cached during analysis
- Ensure you have proper consent from candidates before recording and analyzing interviews

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

