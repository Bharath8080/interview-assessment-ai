import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from google.generativeai import upload_file, get_file
import google.generativeai as genai
import time
import json
from pathlib import Path
import tempfile
from dotenv import load_dotenv
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load API key from environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# Set up the webpage layout
st.set_page_config(
    page_title="Interview Assessment AI",
    page_icon="👔",
    layout="wide"
)

# Add custom styling to make the app look better
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-top: 2rem;
    }
    .score-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .skill-header {
        font-weight: bold;
        font-size: 1.1rem;
    }
    .stTextArea textarea {
        height: 100px;
    }
    .highlight {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# Display the app title
st.markdown("<h1 class='main-header'>AI Interview Assessment Platform</h1>", unsafe_allow_html=True)

# Create the AI agent that will analyze interviews
# @st.cache_resource prevents recreating the agent every time the app refreshes
@st.cache_resource
def initialize_agent():
    return Agent(
        name="Interview Analyzer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        markdown=True,
    )

# Initialize the agent
interview_agent = initialize_agent()

# Define all the categories the AI will use to evaluate candidates
# Each category has a weight (importance) and subcategories for detailed assessment
assessment_categories = {
    "technical_skills": {
        "name": "Technical Skills",
        "weight": 0.30,  # This category is 30% of the total score
        "subcategories": {
            "core_knowledge": "Understanding of domain-specific concepts",
            "problem_solving": "Approach to solving technical problems",
            "coding_skills": "Proficiency in programming languages (if applicable)",
            "tools_technologies": "Familiarity with industry-standard tools"
        }
    },
    "communication_skills": {
        "name": "Communication Skills",
        "weight": 0.20,  # 20% of total score
        "subcategories": {
            "clarity": "Ability to express thoughts clearly",
            "listening": "Understanding and responding appropriately",
            "conciseness": "Being to the point without unnecessary details",
            "nonverbal": "Body language, eye contact, and overall presence"
        }
    },
    "behavioral_skills": {
        "name": "Behavioral & Soft Skills",
        "weight": 0.15,  # 15% of total score
        "subcategories": {
            "leadership": "Leadership potential and teamwork abilities",
            "adaptability": "Flexibility in handling different situations",
            "problem_solving_mindset": "Approach to challenges",
            "emotional_intelligence": "Handling stress, feedback, and collaboration"
        }
    },
    "strengths_weaknesses": {
        "name": "Strengths & Weaknesses",
        "weight": 0.10,  # 10% of total score
        "subcategories": {
            "self_awareness": "Understanding of capabilities and gaps",
            "improvement_mindset": "How weaknesses are addressed"
        }
    },
    "cultural_fit": {
        "name": "Cultural Fit & Attitude",
        "weight": 0.10,  # 10% of total score
        "subcategories": {
            "values_alignment": "Alignment with company values",
            "growth_mindset": "Willingness to learn and improve",
            "work_ethic": "Dedication, reliability, and responsibility"
        }
    },
    "critical_thinking": {
        "name": "Problem-Solving & Critical Thinking",
        "weight": 0.10,  # 10% of total score
        "subcategories": {
            "logical_thinking": "Structured approach to problem-solving",
            "creativity": "Ability to think out of the box"
        }
    },
    "decision_making": {
        "name": "Decision-Making Ability",
        "weight": 0.05,  # 5% of total score
        "subcategories": {
            "analytical_thinking": "Weighing pros and cons before making choices",
            "pressure_handling": "Making sound decisions under stress"
        }
    }
}

# Create a sidebar for setting up the interview parameters
with st.sidebar:
    st.header("Interview Settings")
    
    # Dropdown to select what job role is being interviewed for
    job_role = st.selectbox(
        "Select Job Role",
        ["Software Engineer", "Data Scientist", "Product Manager", "UX Designer", "Marketing Specialist", "Sales Representative", "Customer Support", "Project Manager", "Other"]
    )
    
    # Dropdown to select candidate's experience level
    experience_level = st.selectbox(
        "Experience Level",
        ["Entry Level (0-2 years)", "Mid Level (3-5 years)", "Senior (6-10 years)", "Expert (10+ years)"]
    )
    
    # Allow users to customize how important each category is
    st.subheader("Evaluation Focus")
    custom_weights = {}
    for cat_id, category in assessment_categories.items():
        custom_weights[cat_id] = st.slider(
            f"{category['name']} Importance", 
            min_value=0.0, 
            max_value=1.0, 
            value=float(category['weight']),
            step=0.05,
            format="%.2f"
        )
    
    # Make sure all weights add up to 1.0 (100%)
    total_weight = sum(custom_weights.values())
    if total_weight > 0:
        for cat_id in custom_weights:
            custom_weights[cat_id] = custom_weights[cat_id] / total_weight
    
    st.sidebar.info("Upload a video interview and click 'Analyze Interview' to receive a detailed assessment.")

# Create three tabs to organize the app interface
tab1, tab2, tab3 = st.tabs(["Interview Upload", "Results Dashboard", "Analysis Details"])

# First tab - Upload interview video and set parameters
with tab1:
    # Add a file uploader for interview videos
    video_file = st.file_uploader(
        "Upload an interview video", 
        type=['mp4', 'mov', 'avi'], 
        help="Upload a video interview for AI analysis"
    )
    
    if video_file:
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name
        
        # Display the video in the app
        st.video(video_path, format="video/mp4", start_time=0)
        
        # Add input fields for candidate information
        col1, col2 = st.columns(2)
        with col1:
            candidate_name = st.text_input("Candidate Name (Optional)")
        with col2:
            position_applied = st.text_input("Position Applied For", value=job_role)
        
        # Add a text area for any specific focus points
        specific_questions = st.text_area(
            "Any specific aspects to focus on? (Optional)",
            placeholder="Example: Pay special attention to their explanation of the database design project."
        )
        
        # Add a button to start the analysis
        analyze_button = st.button("🔍 Analyze Interview", type="primary")
        
        if analyze_button:
            try:
                with st.spinner("Processing interview video and generating comprehensive assessment..."):
                    # Upload the video to Google's AI for processing
                    processed_video = upload_file(video_path)
                    
                    # Wait until the video is fully processed
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)
                    
                    # Create a detailed prompt for the AI to analyze the interview
                    analysis_prompt = f"""
                    You are an expert interview assessor with deep experience in talent acquisition and human resources.
                    
                    Analyze the uploaded interview video for a {position_applied} position at {experience_level} experience level.
                    
                    Conduct a comprehensive assessment of the candidate{' ' + candidate_name if candidate_name else ''} and provide:
                    
                    1. Overall impression and summary (100-150 words)
                    2. For each category below, provide:
                       - A score from 0-100
                       - 2-3 specific observations with timestamps
                       - A brief qualitative assessment (30-50 words)
                    
                    Assessment categories:
                    {json.dumps(assessment_categories, indent=2)}
                    
                    For each subcategory, provide a score from 0-100.
                    
                    3. Key strengths (3-5 bullet points)
                    4. Areas for improvement (3-5 bullet points)
                    5. Overall fit for the role (Strong/Moderate/Limited) with justification
                    6. Final score out of 100 based on weighted category scores
                    
                    Additional focus areas to consider:
                    {specific_questions}
                    
                    Format your response as a JSON object with the following structure:
                    {{
                        "summary": "Overall impression summary",
                        "categories": {{
                            "technical_skills": {{
                                "score": 85,
                                "observations": ["Observation 1 (2:15)", "Observation 2 (5:43)"],
                                "assessment": "Brief qualitative assessment",
                                "subcategories": {{
                                    "core_knowledge": 80,
                                    "problem_solving": 85,
                                    "coding_skills": 90,
                                    "tools_technologies": 85
                                }}
                            }},
                            // Other categories following the same pattern
                        }},
                        "strengths": ["Strength 1", "Strength 2", "Strength 3"],
                        "improvements": ["Area 1", "Area 2", "Area 3"],
                        "role_fit": {{
                            "rating": "Strong",
                            "justification": "Justification text"
                        }},
                        "final_score": 82
                    }}
                    
                    Make sure your JSON is valid with proper escaping of quotes and special characters.
                    """
                    
                    # Send the prompt to the AI agent along with the video
                    response = interview_agent.run(analysis_prompt, videos=[processed_video])
                    
                    # Extract the JSON result from the AI's response
                    json_str = response.content
                    # Find JSON content between triple backticks if present
                    if "```json" in json_str:
                        json_str = json_str.split("```json")[1].split("```")[0].strip()
                    elif "```" in json_str:
                        json_str = json_str.split("```")[1].split("```")[0].strip()
                    
                    # Convert the JSON string to a Python dictionary
                    assessment_data = json.loads(json_str)
                    
                    # Save the results to session state so they persist between tabs
                    st.session_state.assessment_data = assessment_data
                    st.session_state.candidate_name = candidate_name
                    st.session_state.position_applied = position_applied
                    
                    # Show success message and guide user to results tab
                    st.success("Interview analysis complete! View the results in the 'Results Dashboard' tab.")
                    
            except Exception as error:
                # Show error message if something goes wrong
                st.error(f"An error occurred during analysis: {error}")
            finally:
                # Clean up the temporary video file
                Path(video_path).unlink(missing_ok=True)
    else:
        # Show info message when no video is uploaded yet
        st.info("Upload an interview video to begin the assessment.")

# Second tab - Dashboard showing assessment results
with tab2:
    # Only show results if analysis has been completed
    if 'assessment_data' in st.session_state:
        assessment_data = st.session_state.assessment_data
        candidate_name = st.session_state.candidate_name if st.session_state.candidate_name else "Candidate"
        position = st.session_state.position_applied
        
        # Create a header section with candidate info and overall score
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"### Interview Assessment: {candidate_name}")
            st.markdown(f"**Position:** {position}")
            st.markdown(f"**Overall Fit:** {assessment_data['role_fit']['rating']}")
        with col2:
            # Create a gauge chart to visually show the final score
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = assessment_data['final_score'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Final Score"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2563EB"},
                    'steps': [
                        {'range': [0, 50], 'color': "#FEE2E2"},  # Red for low scores
                        {'range': [50, 75], 'color': "#FEF3C7"},  # Yellow for medium scores
                        {'range': [75, 100], 'color': "#DCFCE7"}  # Green for high scores
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 85
                    }
                }
            ))
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        
        # Display the overall summary
        st.markdown("### Summary")
        st.markdown(f"<div class='highlight'>{assessment_data['summary']}</div>", unsafe_allow_html=True)
        
        # Show key strengths and areas for improvement side by side
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Key Strengths")
            for strength in assessment_data['strengths']:
                st.markdown(f"✅ {strength}")
        with col2:
            st.markdown("### Areas for Improvement")
            for improvement in assessment_data['improvements']:
                st.markdown(f"🔸 {improvement}")
        
        # Create a radar chart showing scores across all categories
        st.markdown("### Performance by Category")
        
        # Prepare data for the radar chart
        categories = []
        scores = []
        
        for cat_id, category_data in assessment_data['categories'].items():
            category_name = assessment_categories[cat_id]['name']
            categories.append(category_name)
            scores.append(category_data['score'])
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            name='Candidate Score',
            line_color='#3B82F6',
            fillcolor='rgba(59, 130, 246, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Bar chart for subcategories
        st.markdown("### Detailed Skills Breakdown")
        
        # Prepare subcategory data
        subcategory_data = []
        
        for cat_id, category_data in assessment_data['categories'].items():
            category_name = assessment_categories[cat_id]['name']
            if 'subcategories' in category_data:
                for subcat_id, score in category_data['subcategories'].items():
                    subcat_name = assessment_categories[cat_id]['subcategories'][subcat_id]
                    subcategory_data.append({
                        'Category': category_name,
                        'Subcategory': subcat_name,
                        'Score': score
                    })
        
        if subcategory_data:
            subcat_df = pd.DataFrame(subcategory_data)
            
            # Create grouped bar chart for subcategories
            fig = px.bar(
                subcat_df, 
                x='Subcategory', 
                y='Score', 
                color='Category',
                color_discrete_sequence=px.colors.qualitative.Set3,
                title='Subcategory Performance',
                labels={'Score': 'Score (0-100)'},
                height=500
            )
            
            fig.update_layout(
                xaxis={'categoryorder':'total descending'},
                xaxis_tickangle=-45,
                margin=dict(b=100)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Role fit justification
        st.markdown("### Role Fit Analysis")
        st.markdown(f"<div class='score-card' style='background-color: {'#DCFCE7' if assessment_data['role_fit']['rating'] == 'Strong' else '#FEF3C7' if assessment_data['role_fit']['rating'] == 'Moderate' else '#FEE2E2'};'><span class='skill-header'>Rating: {assessment_data['role_fit']['rating']}</span><br>{assessment_data['role_fit']['justification']}</div>", unsafe_allow_html=True)

with tab3:
    if 'assessment_data' in st.session_state:
        assessment_data = st.session_state.assessment_data
        
        st.markdown("### Detailed Category Analysis")
        
        # Create expandable sections for each category
        for cat_id, category_data in assessment_data['categories'].items():
            category_name = assessment_categories[cat_id]['name']
            
            with st.expander(f"{category_name} - Score: {category_data['score']}/100"):
                # Display qualitative assessment
                st.markdown(f"**Assessment:** {category_data['assessment']}")
                
                # Display observations with timestamps
                st.markdown("**Key Observations:**")
                for observation in category_data['observations']:
                    st.markdown(f"- {observation}")
                
                # Display subcategory scores
                if 'subcategories' in category_data:
                    st.markdown("**Skills Breakdown:**")
                    
                    # Create a horizontal bar chart for subcategory scores
                    subcat_names = []
                    subcat_scores = []
                    
                    for subcat_id, score in category_data['subcategories'].items():
                        subcat_name = assessment_categories[cat_id]['subcategories'][subcat_id]
                        subcat_names.append(subcat_name)
                        subcat_scores.append(score)
                    
                    subcat_df = pd.DataFrame({
                        'Subcategory': subcat_names,
                        'Score': subcat_scores
                    })
                    
                    fig = px.bar(
                        subcat_df,
                        x='Score',
                        y='Subcategory',
                        orientation='h',
                        range_x=[0, 100],
                        color='Score',
                        color_continuous_scale=[(0, "red"), (0.5, "yellow"), (1, "green")]
                    )
                    
                    fig.update_layout(height=50 + (len(subcat_names) * 30))
                    st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        st.markdown("### Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate PDF Report"):
                st.info("PDF report generation functionality would be implemented here.")
        
        with col2:
            # Create JSON for download
            json_str = json.dumps(assessment_data, indent=2)
            st.download_button(
                label="Download Raw JSON Data",
                data=json_str,
                file_name=f"interview_assessment_{candidate_name.replace(' ', '_')}.json",
                mime="application/json"
            )
    else:
        st.info("Complete an interview analysis first to view detailed results.")



