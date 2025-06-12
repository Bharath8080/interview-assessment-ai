import os
import streamlit as st
import requests
from dotenv import load_dotenv
import time
import tempfile
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional, Tuple, Any
import hashlib
import re
from pathlib import Path
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Config:
    """Application configuration management"""
    
    # API Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
    
    # AssemblyAI endpoints
    TRANSCRIPT_ENDPOINT = "https://api.assemblyai.com/v2/transcript"
    UPLOAD_ENDPOINT = 'https://api.assemblyai.com/v2/upload'
    CHUNK_SIZE = 5242880
    
    # File constraints
    MAX_FILE_SIZE_MB = 100
    ALLOWED_AUDIO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mp3', '.wav', '.m4a', '.webm']
    
    # Model configuration
    GEMINI_MODEL = 'gemini-2.5-flash-preview-05-20'  # Changed to stable version
    MAX_RETRIES = 3
    RETRY_DELAY = 2

class SecurityManager:
    """Handle security-related operations"""
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal attacks"""
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        return filename[:100]  # Limit filename length
    
    @staticmethod
    def validate_file_size(file_size: int) -> bool:
        """Validate file size constraints"""
        max_size = Config.MAX_FILE_SIZE_MB * 1024 * 1024
        return file_size <= max_size
    
    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        """Validate file extension"""
        extension = Path(filename).suffix.lower()
        return extension in Config.ALLOWED_AUDIO_EXTENSIONS

class AssessmentCategories:
    """Define assessment categories and weights"""
    
    CATEGORIES = {
        "technical_skills": {
            "name": "Technical Skills",
            "weight": 0.30,
            "subcategories": {
                "core_knowledge": "Understanding of domain-specific concepts",
                "problem_solving": "Approach to solving technical problems",
                "coding_skills": "Proficiency in programming languages",
                "tools_technologies": "Familiarity with industry-standard tools"
            }
        },
        "communication_skills": {
            "name": "Communication Skills",
            "weight": 0.20,
            "subcategories": {
                "clarity": "Ability to express thoughts clearly",
                "listening": "Understanding and responding appropriately",
                "conciseness": "Being to the point without unnecessary details",
                "nonverbal": "Body language and overall presence"
            }
        },
        "behavioral_skills": {
            "name": "Behavioral & Soft Skills",
            "weight": 0.15,
            "subcategories": {
                "leadership": "Leadership potential and teamwork abilities",
                "adaptability": "Flexibility in handling different situations",
                "problem_solving_mindset": "Approach to challenges",
                "emotional_intelligence": "Handling stress and feedback"
            }
        },
        "strengths_weaknesses": {
            "name": "Strengths & Weaknesses",
            "weight": 0.10,
            "subcategories": {
                "self_awareness": "Understanding of capabilities and gaps",
                "improvement_mindset": "How weaknesses are addressed"
            }
        },
        "cultural_fit": {
            "name": "Cultural Fit & Attitude",
            "weight": 0.10,
            "subcategories": {
                "values_alignment": "Alignment with company values",
                "growth_mindset": "Willingness to learn and improve",
                "work_ethic": "Dedication and responsibility"
            }
        },
        "critical_thinking": {
            "name": "Problem-Solving & Critical Thinking",
            "weight": 0.10,
            "subcategories": {
                "logical_thinking": "Structured approach to problem-solving",
                "creativity": "Ability to think out of the box"
            }
        },
        "decision_making": {
            "name": "Decision-Making Ability",
            "weight": 0.05,
            "subcategories": {
                "analytical_thinking": "Weighing pros and cons",
                "pressure_handling": "Making sound decisions under stress"
            }
        }
    }

class TranscriptionService:
    """Handle audio transcription using AssemblyAI"""
    
    def __init__(self):
        self.api_key = Config.ASSEMBLYAI_API_KEY
        self.headers = {'authorization': self.api_key}
    
    def _read_file_chunks(self, filepath: str):
        """Read file in chunks for upload"""
        with open(filepath, 'rb') as file:
            while True:
                data = file.read(Config.CHUNK_SIZE)
                if not data:
                    break
                yield data
    
    def upload_file(self, filepath: str) -> Optional[str]:
        """Upload file to AssemblyAI and return audio URL"""
        try:
            response = requests.post(
                Config.UPLOAD_ENDPOINT,
                headers=self.headers,
                data=self._read_file_chunks(filepath),
                timeout=300  # 5 minute timeout
            )
            response.raise_for_status()
            return response.json().get('upload_url')
        except requests.exceptions.RequestException as e:
            logger.error(f"File upload failed: {e}")
            raise Exception(f"Failed to upload file: {str(e)}")
    
    def start_transcription(self, audio_url: str) -> Optional[str]:
        """Start transcription process and return transcript ID"""
        try:
            transcript_request = {
                'audio_url': audio_url,
                'language_detection': True,
                'punctuate': True,
                'format_text': True
            }
            
            response = requests.post(
                Config.TRANSCRIPT_ENDPOINT,
                json=transcript_request,
                headers={**self.headers, 'content-type': 'application/json'},
                timeout=30
            )
            response.raise_for_status()
            return response.json().get('id')
        except requests.exceptions.RequestException as e:
            logger.error(f"Transcription start failed: {e}")
            raise Exception(f"Failed to start transcription: {str(e)}")
    
    def get_transcription_status(self, transcript_id: str) -> Optional[Dict]:
        """Get transcription status and result"""
        try:
            polling_endpoint = f"{Config.TRANSCRIPT_ENDPOINT}/{transcript_id}"
            response = requests.get(polling_endpoint, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Status check failed: {e}")
            raise Exception(f"Failed to check transcription status: {str(e)}")
    
    def transcribe_file(self, filepath: str, progress_callback=None) -> str:
        """Complete transcription workflow"""
        try:
            # Upload file
            if progress_callback:
                progress_callback("Uploading file...")
            audio_url = self.upload_file(filepath)
            
            # Start transcription
            if progress_callback:
                progress_callback("Starting transcription...")
            transcript_id = self.start_transcription(audio_url)
            
            # Poll for completion
            max_wait_time = 1800  # 30 minutes
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                if progress_callback:
                    progress_callback("Processing audio...")
                
                status_data = self.get_transcription_status(transcript_id)
                status = status_data.get('status')
                
                if status == 'completed':
                    return status_data.get('text', '')
                elif status == 'error':
                    error_msg = status_data.get('error', 'Unknown transcription error')
                    raise Exception(f"Transcription failed: {error_msg}")
                
                time.sleep(Config.RETRY_DELAY)
            
            raise Exception("Transcription timed out")
            
        except Exception as e:
            logger.error(f"Transcription workflow failed: {e}")
            raise

class AIAnalyzer:
    """Handle AI-powered interview analysis"""
    
    def __init__(self):
        if not Config.GOOGLE_API_KEY:
            raise Exception("Google API key is not configured")
        
        try:
            genai.configure(api_key=Config.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
            logger.info(f"Initialized Gemini model: {Config.GEMINI_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {str(e)}")
            raise Exception(f"Failed to initialize AI model: {str(e)}")
    
    def _create_analysis_prompt(self, transcript: str, job_role: str, 
                             experience_level: str, candidate_name: str = "") -> str:
        """Create structured prompt for AI analysis"""
        return f"""
        You are an expert interview assessor with deep experience in talent acquisition and human resources.
        
        Analyze the following interview transcript for a {job_role} position at {experience_level} experience level.
        
        Candidate: {candidate_name if candidate_name else "Not specified"}
        
        Interview Transcript:
        {transcript}
        
        Conduct a comprehensive assessment and provide:
        
        1. Overall impression and summary (100-150 words)
        2. For each category below, provide:
           - A score from 0-100
           - 2-3 specific observations
           - A brief qualitative assessment (30-50 words)
        
        Assessment categories:
        {json.dumps(AssessmentCategories.CATEGORIES, indent=2)}
        
        For each subcategory, provide a score from 0-100.
        
        3. Key strengths (3-5 bullet points)
        4. Areas for improvement (3-5 bullet points)
        5. Overall fit for the role (Strong/Moderate/Limited) with justification
        6. Final score out of 100 based on weighted category scores
        
        Format your response as a JSON object with the following structure:
        {{
            "summary": "Overall impression summary",
            "categories": {{
                "technical_skills": {{
                    "score": 85,
                    "observations": ["Observation 1", "Observation 2"],
                    "assessment": "Brief qualitative assessment",
                    "subcategories": {{
                        "core_knowledge": 80,
                        "problem_solving": 85,
                        "coding_skills": 90,
                        "tools_technologies": 85
                    }}
                }}
            }},
            "strengths": ["Strength 1", "Strength 2", "Strength 3"],
            "improvements": ["Area 1", "Area 2", "Area 3"],
            "role_fit": {{
                "rating": "Strong",
                "justification": "Justification text"
            }},
            "final_score": 82
        }}
        """
    
    def analyze_interview(self, transcript: str, job_role: str, 
                         experience_level: str, candidate_name: str = "") -> Optional[Dict]:
        """Analyze interview transcript and return structured assessment"""
        try:
            if not self.model:
                raise Exception("AI model not initialized")
            
            prompt = self._create_analysis_prompt(transcript, job_role, experience_level, candidate_name)
            logger.info("Generated analysis prompt")
            
            for attempt in range(Config.MAX_RETRIES):
                try:
                    logger.info(f"Attempting analysis (attempt {attempt + 1}/{Config.MAX_RETRIES})")
                    response = self.model.generate_content(prompt)
                    
                    if not response or not response.candidates:
                        raise Exception("Empty response from AI model")
                    
                    # Get the first candidate's content
                    content = response.candidates[0].content
                    if not content or not content.parts:
                        raise Exception("No content parts in response")
                    
                    # Get the text from the first part
                    analysis_text = content.parts[0].text
                    if not analysis_text:
                        raise Exception("Empty text in response")
                    
                    logger.info("Received response from AI model")
                    
                    # Extract JSON from response
                    if "```json" in analysis_text:
                        analysis_text = analysis_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in analysis_text:
                        analysis_text = analysis_text.split("```")[1].split("```")[0].strip()
                    
                    # Parse and validate JSON
                    assessment_data = json.loads(analysis_text)
                    logger.info("Successfully parsed assessment data")
                    
                    # Add metadata
                    assessment_data['metadata'] = {
                        'analysis_timestamp': datetime.now().isoformat(),
                        'job_role': job_role,
                        'experience_level': experience_level,
                        'candidate_name': candidate_name,
                        'model_used': Config.GEMINI_MODEL
                    }
                    
                    return assessment_data
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing failed on attempt {attempt + 1}: {e}")
                    if attempt == Config.MAX_RETRIES - 1:
                        raise Exception("Failed to parse AI response after multiple attempts")
                    time.sleep(Config.RETRY_DELAY)
                    
        except Exception as e:
            logger.error(f"Interview analysis failed: {str(e)}", exc_info=True)
            if "quota" in str(e).lower():
                raise Exception("API quota exceeded. Please check your billing or try again later.")
            raise Exception(f"Analysis failed: {str(e)}")

class ReportGenerator:
    """Generate assessment reports and visualizations"""
    
    @staticmethod
    def create_score_gauge(score: int, title: str = "Final Score") -> go.Figure:
        """Create gauge chart for scores"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2563EB"},
                'steps': [
                    {'range': [0, 50], 'color': "#FEE2E2"},
                    {'range': [50, 75], 'color': "#FEF3C7"},
                    {'range': [75, 100], 'color': "#DCFCE7"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 85
                }
            }
        ))
        fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
        return fig
    
    @staticmethod
    def create_radar_chart(categories: Dict, assessment_data: Dict) -> go.Figure:
        """Create radar chart for category performance"""
        category_names = []
        scores = []
        
        for cat_id, category_data in assessment_data['categories'].items():
            if cat_id in categories:
                category_names.append(categories[cat_id]['name'])
                scores.append(category_data['score'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=category_names,
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
        return fig
    
    @staticmethod
    def create_subcategory_chart(category_data: Dict, category_info: Dict) -> go.Figure:
        """Create horizontal bar chart for subcategories"""
        if 'subcategories' not in category_data:
            return None
        
        subcat_names = []
        subcat_scores = []
        
        for subcat_id, score in category_data['subcategories'].items():
            if subcat_id in category_info['subcategories']:
                subcat_name = category_info['subcategories'][subcat_id]
                subcat_names.append(subcat_name)
                subcat_scores.append(score)
        
        df = pd.DataFrame({
            'Subcategory': subcat_names,
            'Score': subcat_scores
        })
        
        fig = px.bar(
            df,
            x='Score',
            y='Subcategory',
            orientation='h',
            range_x=[0, 100],
            color='Score',
            color_continuous_scale=[(0, "red"), (0.5, "yellow"), (1, "green")]
        )
        
        fig.update_layout(height=50 + (len(subcat_names) * 30))
        return fig

class SessionManager:
    """Manage Streamlit session state"""
    
    @staticmethod
    def initialize_session():
        """Initialize session state variables"""
        default_values = {
            'assessment_data': None,
            'transcript': None,
            'candidate_name': '',
            'position_applied': '',
            'analysis_complete': False,
            'session_id': str(uuid.uuid4())
        }
        
        for key, value in default_values.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def clear_session():
        """Clear session data for new analysis"""
        keys_to_clear = ['assessment_data', 'transcript', 'analysis_complete']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="AI Interview Assessment Platform",
        page_icon="👔",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling in both light and dark modes
    st.markdown("""
    <style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Card and container styling */
    .highlight {
        background-color: var(--background-color, rgba(59, 130, 246, 0.1));
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid var(--accent-color, #3B82F6);
        margin: 1rem 0;
        color: var(--text-color, inherit);
    }
    
    .score-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid var(--border-color, rgba(59, 130, 246, 0.2));
        background-color: var(--card-background, rgba(59, 130, 246, 0.05));
        color: var(--text-color, inherit);
    }
    
    .skill-header {
        font-weight: bold;
        font-size: 1.1rem;
        color: var(--text-color, inherit);
    }
    
    .metric-container {
        background-color: var(--metric-background, rgba(59, 130, 246, 0.05));
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid var(--border-color, rgba(59, 130, 246, 0.2));
        color: var(--text-color, inherit);
    }
    
    /* Preparation section styling */
    .prep-card {
        background-color: var(--prep-background, rgba(59, 130, 246, 0.05));
        border: 1px solid var(--border-color, rgba(59, 130, 246, 0.2));
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        color: var(--text-color, inherit);
    }
    
    .prep-header {
        color: var(--accent-color, #3B82F6);
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: var(--accent-color, #3B82F6);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--accent-color, #3B82F6);
        color: white;
    }
    
    .stButton > button:hover {
        background-color: var(--accent-hover, #2563EB);
    }
    
    /* Score colors */
    .score-high {
        background-color: var(--score-high, rgba(34, 197, 94, 0.2)) !important;
        border-color: var(--score-high-border, rgba(34, 197, 94, 0.4)) !important;
    }
    
    .score-medium {
        background-color: var(--score-medium, rgba(234, 179, 8, 0.2)) !important;
        border-color: var(--score-medium-border, rgba(234, 179, 8, 0.4)) !important;
    }
    
    .score-low {
        background-color: var(--score-low, rgba(239, 68, 68, 0.2)) !important;
        border-color: var(--score-low-border, rgba(239, 68, 68, 0.4)) !important;
    }
    
    /* Executive Summary specific styling */
    .executive-summary {
        background-color: var(--summary-background, rgba(59, 130, 246, 0.05));
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border: 1px solid var(--summary-border, rgba(59, 130, 246, 0.2));
        color: var(--text-color, inherit);
        line-height: 1.6;
        font-size: 1.1rem;
    }
    
    .executive-summary-header {
        color: var(--header-text-color, #1F2937);
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--summary-border, rgba(59, 130, 246, 0.2));
        padding-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .executive-summary-header .emoji {
        font-size: 1.8rem;
        filter: var(--emoji-filter, none);
    }
    
    /* Dark mode adjustments */
    @media (prefers-color-scheme: dark) {
        :root {
            --background-color: rgba(59, 130, 246, 0.15);
            --accent-color: #60A5FA;
            --accent-hover: #93C5FD;
            --text-color: #E5E7EB;
            --border-color: rgba(59, 130, 246, 0.3);
            --card-background: rgba(59, 130, 246, 0.1);
            --metric-background: rgba(59, 130, 246, 0.1);
            --prep-background: rgba(59, 130, 246, 0.1);
            --score-high: rgba(34, 197, 94, 0.3);
            --score-high-border: rgba(34, 197, 94, 0.5);
            --score-medium: rgba(234, 179, 8, 0.3);
            --score-medium-border: rgba(234, 179, 8, 0.5);
            --score-low: rgba(239, 68, 68, 0.3);
            --score-low-border: rgba(239, 68, 68, 0.5);
            --summary-background: rgba(59, 130, 246, 0.15);
            --summary-border: rgba(59, 130, 246, 0.3);
            --header-text-color: #E5E7EB;
            --emoji-filter: brightness(1.2);
        }
    }
    
    /* Light mode adjustments */
    @media (prefers-color-scheme: light) {
        :root {
            --background-color: rgba(59, 130, 246, 0.1);
            --accent-color: #3B82F6;
            --accent-hover: #2563EB;
            --text-color: #1F2937;
            --border-color: rgba(59, 130, 246, 0.2);
            --card-background: rgba(59, 130, 246, 0.05);
            --metric-background: rgba(59, 130, 246, 0.05);
            --prep-background: rgba(59, 130, 246, 0.05);
            --score-high: rgba(34, 197, 94, 0.2);
            --score-high-border: rgba(34, 197, 94, 0.4);
            --score-medium: rgba(234, 179, 8, 0.2);
            --score-medium-border: rgba(234, 179, 8, 0.4);
            --score-low: rgba(239, 68, 68, 0.2);
            --score-low-border: rgba(239, 68, 68, 0.4);
            --summary-background: rgba(59, 130, 246, 0.05);
            --summary-border: rgba(59, 130, 246, 0.2);
            --header-text-color: #1F2937;
            --emoji-filter: none;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar with configuration options"""
    with st.sidebar:
        st.header("🎯 Interview Settings")
        
        # Job role selection
        job_role = st.selectbox(
            "Select Job Role",
            ["Frontend Developer", "Backend Developer", "Full Stack Developer", "Mobile Developer", 
             "DevOps Engineer", "Software Architect", "Security Engineer", "Data Engineer", 
             "Video Game Developer", "Quality Engineering", "Cloud Engineer", "Systems Engineer", 
             "Machine Learning Engineer", "Other"],
            help="Select the role for which the candidate is being interviewed"
        )
        
        # Experience level selection
        experience_level = st.selectbox(
            "Experience Level",
            ["Entry Level (0-2 years)", "Mid Level (3-5 years)", 
             "Senior (6-10 years)", "Expert (10+ years)"],
            help="Select the candidate's experience level"
        )
        
        # API Keys section
        with st.expander("🔑 API Configuration", expanded=not (Config.GOOGLE_API_KEY and Config.ASSEMBLYAI_API_KEY)):
            st.markdown("**Required API Keys:**")
            st.markdown("- [Google AI Studio](https://makersuite.google.com/app/apikey)")
            st.markdown("- [AssemblyAI](https://www.assemblyai.com/)")
            
            google_api_key = st.text_input(
                "Google API Key:", 
                type="password", 
                value=Config.GOOGLE_API_KEY or "",
                help="Required for AI analysis"
            )
            assembly_api_key = st.text_input(
                "AssemblyAI API Key:", 
                type="password", 
                value=Config.ASSEMBLYAI_API_KEY or "",
                help="Required for audio transcription"
            )
            
            # Update configuration
            if google_api_key:
                Config.GOOGLE_API_KEY = google_api_key
                os.environ['GOOGLE_API_KEY'] = google_api_key
            if assembly_api_key:
                Config.ASSEMBLYAI_API_KEY = assembly_api_key
                os.environ['ASSEMBLYAI_API_KEY'] = assembly_api_key
        
        # System status
        st.markdown("---")
        st.markdown("### System Status")
        
        google_status = "✅ Connected" if Config.GOOGLE_API_KEY else "❌ Not configured"
        assembly_status = "✅ Connected" if Config.ASSEMBLYAI_API_KEY else "❌ Not configured"
        
        st.markdown(f"**Google AI:** {google_status}")
        st.markdown(f"**AssemblyAI:** {assembly_status}")
        
        return job_role, experience_level

def render_preparation_tab():
    """Render the interview preparation tab"""
    st.markdown("### 🎯 Interview Preparation")
    
    # Role-specific preparation
    selected_role = st.selectbox(
        "Select Target Role",
        ["Frontend Developer", "Backend Developer", "Full Stack Developer", "Mobile Developer", 
         "DevOps Engineer", "Software Architect", "Security Engineer", "Data Engineer", 
         "Video Game Developer", "Quality Engineering", "Cloud Engineer", "Systems Engineer", 
         "Machine Learning Engineer", "Other"]
    )
    
    # Preparation sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📚 Technical Preparation")
        
        # Technical topics
        tech_topics = {
            "Frontend Developer": [
                "HTML/CSS",
                "JavaScript",
                "React",
                "Angular",
                "Web Development Frameworks"
            ],
            "Backend Developer": [
                "Python",
                "Java",
                "Node.js",
                "Database Design",
                "API Development"
            ],
            "Full Stack Developer": [
                "JavaScript",
                "Python",
                "React",
                "Node.js",
                "Database Design"
            ],
            "Mobile Developer": [
                "Android Development",
                "iOS Development",
                "Mobile App Design",
                "Cross-Platform Development",
                "Mobile App Testing"
            ],
            "DevOps Engineer": [
                "Docker",
                "Kubernetes",
                "CI/CD",
                "Monitoring and Logging",
                "Infrastructure as Code"
            ],
            "Software Architect": [
                "System Design",
                "Microservices Architecture",
                "Cloud Architecture",
                "Software Development Practices",
                "Design Patterns"
            ],
            "Security Engineer": [
                "Network Security",
                "Web Application Security",
                "Cryptography",
                "Security Audits",
                "Incident Response"
            ],
            "Data Engineer": [
                "Big Data Technologies",
                "Data Processing",
                "Data Storage",
                "Data Visualization",
                "Data Engineering Practices"
            ],
            "Video Game Developer": [
                "Game Design",
                "Unity",
                "Unreal Engine",
                "Game Development Tools",
                "Game Programming"
            ],
            "Quality Engineering": [
                "Testing Frameworks",
                "Test Automation",
                "Continuous Integration",
                "Performance Testing",
                "Agile Testing"
            ],
            "Cloud Engineer": [
                "Cloud Platforms",
                "Serverless Architecture",
                "Containerization",
                "Cloud Security",
                "Cloud Management"
            ],
            "Systems Engineer": [
                "System Architecture",
                "Distributed Systems",
                "Network Configuration",
                "System Integration",
                "System Administration"
            ],
            "Machine Learning Engineer": [
                "Machine Learning Fundamentals",
                "Deep Learning",
                "Data Science",
                "Python Programming",
                "Model Training and Evaluation"
            ],
            # Add more roles and their topics
        }
        
        if selected_role in tech_topics:
            for topic in tech_topics[selected_role]:
                with st.expander(topic):
                    st.markdown(f"**Key Areas to Focus On:**")
                    st.markdown(f"- {topic} fundamentals")
                    st.markdown(f"- Recent developments in {topic}")
                    st.markdown(f"- Practical applications of {topic}")
        
        # Practice questions
        st.markdown("#### 💡 Practice Questions")
        if st.button("Generate Practice Questions"):
            # Generate role-specific questions
            st.markdown("**Sample Questions:**")
            st.markdown("1. Tell me about your experience with...")
            st.markdown("2. How would you approach...")
            st.markdown("3. What's your strategy for...")
    
    with col2:
        st.markdown("#### 🎯 Behavioral Preparation")
        
        # Common behavioral questions
        st.markdown("**Common Behavioral Questions:**")
        behavioral_questions = [
            "Tell me about a challenging project you worked on",
            "Describe a situation where you had to work with a difficult team member",
            "How do you handle tight deadlines?",
            "Tell me about a time you had to make a difficult decision"
        ]
        
        for question in behavioral_questions:
            with st.expander(question):
                st.markdown("**STAR Method Response Structure:**")
                st.markdown("- **Situation:** Describe the context")
                st.markdown("- **Task:** What was your responsibility")
                st.markdown("- **Action:** What did you do")
                st.markdown("- **Result:** What was the outcome")
        
        # Interview tips
        st.markdown("#### 💪 Interview Tips")
        tips = [
            "Research the company thoroughly",
            "Prepare questions for the interviewer",
            "Practice your responses out loud",
            "Dress appropriately for the role",
            "Arrive early and be prepared"
        ]
        
        for tip in tips:
            st.markdown(f"- {tip}")
    
    # Mock interview section
    st.markdown("### 🎥 Mock Interview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Record Mock Interview")
        if st.button("Start Recording"):
            st.info("Recording functionality would be implemented here")
    
    with col2:
        st.markdown("#### Upload Mock Interview")
        uploaded_file = st.file_uploader(
            "Upload your mock interview recording",
            type=['mp4', 'mov', 'avi', 'mp3', 'wav', 'm4a', 'webm']
        )
        if uploaded_file:
            st.success("File uploaded successfully!")

def render_upload_tab():
    """Render the interview upload tab"""
    st.markdown("### 📁 Upload Interview Recording")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an interview recording",
        type=['mp4', 'mov', 'avi', 'mp3', 'wav', 'm4a', 'webm'],
        help=f"Supported formats: {', '.join(Config.ALLOWED_AUDIO_EXTENSIONS)}\nMax size: {Config.MAX_FILE_SIZE_MB}MB"
    )
    
    if uploaded_file:
        # Validate file
        if not SecurityManager.validate_file_size(uploaded_file.size):
            st.error(f"File size exceeds {Config.MAX_FILE_SIZE_MB}MB limit")
            return None, None, None
        
        if not SecurityManager.validate_file_extension(uploaded_file.name):
            st.error("Unsupported file format")
            return None, None, None
        
        # Display file info
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.success(f"✅ File uploaded: {uploaded_file.name} ({file_size_mb:.1f}MB)")
        
        # Show video preview if it's a video file
        if uploaded_file.type.startswith('video'):
            st.video(uploaded_file)
        
        # Candidate information form
        with st.form("candidate_info_form"):
            st.markdown("#### Interview Information")
            
            col1, col2 = st.columns(2)
            with col1:
                candidate_name = st.text_input("Candidate Name (Optional)", placeholder="Enter candidate's name")
            with col2:
                position_applied = st.text_input("Position Applied For", value=st.session_state.get('job_role', ''))
            
            interview_notes = st.text_area(
                "Interview Notes (Optional)",
                placeholder="Any specific aspects to focus on or additional context...",
                height=100
            )
            
            # Analysis options
            st.markdown("#### Analysis Options")
            include_sentiment = st.checkbox("Include sentiment analysis", value=True)
            include_keywords = st.checkbox("Extract key topics", value=True)
            include_improvement = st.checkbox("Generate improvement suggestions", value=True)
            
            submitted = st.form_submit_button("🚀 Start Analysis", type="primary")
            
            if submitted:
                if not Config.GOOGLE_API_KEY:
                    st.error("Please configure Google API key in the sidebar")
                    return None, None, None
                if not Config.ASSEMBLYAI_API_KEY:
                    st.error("Please configure AssemblyAI API key in the sidebar")
                    return None, None, None
                
                return uploaded_file, candidate_name, position_applied
    
    else:
        st.info("👆 Upload an interview recording to begin the assessment")
    
    return None, None, None

def process_interview(uploaded_file, candidate_name: str, position_applied: str, 
                     job_role: str, experience_level: str):
    """Process the uploaded interview file"""
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name
    
    try:
        # Initialize services
        transcription_service = TranscriptionService()
        ai_analyzer = AIAnalyzer()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(message: str, progress: float = None):
            status_text.text(message)
            if progress:
                progress_bar.progress(progress)
        
        # Step 1: Transcription
        update_progress("🎵 Transcribing audio...", 0.2)
        
        def transcription_callback(message):
            update_progress(f"🎵 {message}", 0.4)
        
        transcript = transcription_service.transcribe_file(temp_path, transcription_callback)
        
        if not transcript.strip():
            st.error("No speech detected in the audio. Please check the file quality.")
            return
        
        # Step 2: AI Analysis
        update_progress("🤖 Analyzing interview...", 0.6)
        
        assessment_data = ai_analyzer.analyze_interview(
            transcript, position_applied, experience_level, candidate_name
        )
        
        # Step 3: Save results
        update_progress("💾 Saving results...", 0.9)
        
        # Store in session state
        st.session_state.assessment_data = assessment_data
        st.session_state.transcript = transcript
        st.session_state.candidate_name = candidate_name
        st.session_state.position_applied = position_applied
        st.session_state.analysis_complete = True
        
        # Complete
        progress_bar.progress(1.0)
        status_text.text("✅ Analysis complete!")
        
        st.success("🎉 Interview analysis completed successfully!")
        st.balloons()
        
        # Auto-switch to results tab
        st.info("📊 Check the 'Results Dashboard' tab to view your assessment results.")
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        logger.error(f"Interview processing failed: {e}")
    
    finally:
        # Cleanup temporary file
        try:
            os.unlink(temp_path)
        except:
            pass

def render_results_dashboard():
    """Render the results dashboard tab"""
    if not st.session_state.get('analysis_complete') or not st.session_state.get('assessment_data'):
        st.info("📝 Complete an interview analysis first to view results.")
        return
    
    assessment_data = st.session_state.assessment_data
    candidate_name = st.session_state.candidate_name or "Candidate"
    position = st.session_state.position_applied
    
    # Header section
    st.markdown(f"### 📊 Assessment Results: {candidate_name}")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"**Position:** {position}")
        st.markdown(f"**Overall Fit:** {assessment_data['role_fit']['rating']}")
        st.markdown(f"**Analysis Date:** {datetime.now().strftime('%B %d, %Y')}")
    
    with col2:
        # Final score gauge
        fig = ReportGenerator.create_score_gauge(assessment_data['final_score'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Quick metrics
        st.markdown("### Quick Stats")
        st.metric("Final Score", f"{assessment_data['final_score']}/100")
        
        # Calculate average category score
        avg_score = sum(cat['score'] for cat in assessment_data['categories'].values()) / len(assessment_data['categories'])
        st.metric("Avg Category", f"{avg_score:.0f}/100")
    
    # Summary section with new styling
    st.markdown("""
    <div class='executive-summary'>
        <div class='executive-summary-header'>
            <span class='emoji'>📋</span>
            <span>Executive Summary</span>
        </div>
        {summary}
    </div>
    """.format(summary=assessment_data['summary']), unsafe_allow_html=True)
    
    # Strengths and improvements
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Key Strengths")
        for i, strength in enumerate(assessment_data['strengths'], 1):
            st.markdown(f"{i}. {strength}")
    
    with col2:
        st.markdown("### 🎯 Areas for Improvement")
        for i, improvement in enumerate(assessment_data['improvements'], 1):
            st.markdown(f"{i}. {improvement}")
    
    # Performance radar chart
    st.markdown("### 📈 Performance by Category")
    radar_fig = ReportGenerator.create_radar_chart(AssessmentCategories.CATEGORIES, assessment_data)
    st.plotly_chart(radar_fig, use_container_width=True)
    
    # Category breakdown
    st.markdown("### 📊 Category Breakdown")
    
    # Create columns for category cards
    categories_list = list(assessment_data['categories'].items())
    
    for i in range(0, len(categories_list), 3):
        cols = st.columns(3)
        
        for j, col in enumerate(cols):
            if i + j < len(categories_list):
                cat_id, cat_data = categories_list[i + j]
                category_name = AssessmentCategories.CATEGORIES[cat_id]['name']
                
                with col:
                    score = cat_data['score']
                    color = "#DCFCE7" if score >= 75 else "#FEF3C7" if score >= 50 else "#FEE2E2"
                    
                    st.markdown(f"""
                    <div class='score-card' style='background-color: {color};'>
                        <div class='skill-header'>{category_name}</div>
                        <div style='font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;'>{score}/100</div>
                        <div style='font-size: 0.9rem;'>{cat_data['assessment']}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Role fit analysis
    st.markdown("### 🎯 Role Fit Analysis")
    fit_rating = assessment_data['role_fit']['rating']
    fit_color = "#DCFCE7" if fit_rating == "Strong" else "#FEF3C7" if fit_rating == "Moderate" else "#FEE2E2"
    
    st.markdown(f"""
    <div class='score-card' style='background-color: {fit_color};'>
        <div class='skill-header'>Overall Fit: {fit_rating}</div>
        <div style='margin-top: 0.5rem;'>{assessment_data['role_fit']['justification']}</div>
    </div>
    """, unsafe_allow_html=True)

def render_detailed_analysis():
    """Render the detailed analysis tab"""
    if not st.session_state.get('analysis_complete') or not st.session_state.get('assessment_data'):
        st.info("📝 Complete an interview analysis first to view detailed analysis.")
        return
    
    assessment_data = st.session_state.assessment_data
    
    st.markdown("### 🔍 Detailed Category Analysis")
    
    # Create tabs for each category
    category_tabs = st.tabs([AssessmentCategories.CATEGORIES[cat_id]['name'] 
                           for cat_id in assessment_data['categories'].keys()])
    
    for i, (cat_id, category_data) in enumerate(assessment_data['categories'].items()):
        with category_tabs[i]:
            category_info = AssessmentCategories.CATEGORIES[cat_id]
            category_name = category_info['name']
            
            # Category overview
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"#### {category_name}")
                st.markdown(f"**Score: {category_data['score']}/100**")
                st.markdown(f"**Weight in Final Score: {category_info['weight']*100}%**")
                
                st.markdown("**Assessment:**")
                st.markdown(category_data['assessment'])
            
            with col2:
                # Category score gauge
                fig = ReportGenerator.create_score_gauge(category_data['score'], f"{category_name} Score")
                st.plotly_chart(fig, use_container_width=True)
            
            # Key observations
            st.markdown("#### 🔍 Key Observations")
            for j, observation in enumerate(category_data['observations'], 1):
                st.markdown(f"{j}. {observation}")
            
            # Subcategory breakdown
            if 'subcategories' in category_data:
                st.markdown("#### 📊 Skills Breakdown")
                subcat_fig = ReportGenerator.create_subcategory_chart(category_data, category_info)
                if subcat_fig:
                    st.plotly_chart(subcat_fig, use_container_width=True)
                
                # Subcategory details
                st.markdown("#### 📋 Subcategory Details")
                for subcat_id, score in category_data['subcategories'].items():
                    if subcat_id in category_info['subcategories']:
                        subcat_name = category_info['subcategories'][subcat_id]
                        color = "#DCFCE7" if score >= 75 else "#FEF3C7" if score >= 50 else "#FEE2E2"
                        
                        st.markdown(f"""
                        <div class='metric-container' style='background-color: {color};'>
                            <strong>{subcat_name}</strong><br>
                            Score: {score}/100
                        </div>
                        """, unsafe_allow_html=True)
    
    # Transcript section
    st.markdown("---")
    st.markdown("### 📝 Interview Transcript")
    
    with st.expander("View Full Transcript", expanded=False):
        if st.session_state.get('transcript'):
            st.text_area("", st.session_state.transcript, height=300, disabled=True)
        else:
            st.info("Transcript not available")
    
    # Export section
    st.markdown("---")
    st.markdown("### 📄 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # JSON export
        json_data = json.dumps(assessment_data, indent=2)
        st.download_button(
            label="📄 Download JSON Report",
            data=json_data,
            file_name=f"interview_assessment_{st.session_state.candidate_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # CSV export for scores
        scores_data = []
        for cat_id, cat_data in assessment_data['categories'].items():
            category_name = AssessmentCategories.CATEGORIES[cat_id]['name']
            scores_data.append({
                'Category': category_name,
                'Score': cat_data['score'],
                'Assessment': cat_data['assessment']
            })
        
        df = pd.DataFrame(scores_data)
        csv_data = df.to_csv(index=False)
        
        st.download_button(
            label="📊 Download CSV Data",
            data=csv_data,
            file_name=f"interview_scores_{st.session_state.candidate_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col3:
        # Summary report
        summary_report = f"""
INTERVIEW ASSESSMENT REPORT
==========================

Candidate: {st.session_state.candidate_name}
Position: {st.session_state.position_applied}
Date: {datetime.now().strftime('%B %d, %Y')}
Final Score: {assessment_data['final_score']}/100

SUMMARY
-------
{assessment_data['summary']}

ROLE FIT
--------
Rating: {assessment_data['role_fit']['rating']}
{assessment_data['role_fit']['justification']}

KEY STRENGTHS
-------------
{chr(10).join([f"• {strength}" for strength in assessment_data['strengths']])}

AREAS FOR IMPROVEMENT
--------------------
{chr(10).join([f"• {improvement}" for improvement in assessment_data['improvements']])}

CATEGORY SCORES
---------------
{chr(10).join([f"{AssessmentCategories.CATEGORIES[cat_id]['name']}: {cat_data['score']}/100" for cat_id, cat_data in assessment_data['categories'].items()])}
"""
        
        st.download_button(
            label="📋 Download Summary Report",
            data=summary_report,
            file_name=f"interview_summary_{st.session_state.candidate_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

def render_analytics_tab():
    """Render analytics and insights tab"""
    if not st.session_state.get('analysis_complete') or not st.session_state.get('assessment_data'):
        st.info("📝 Complete an interview analysis first to view analytics.")
        return
    
    assessment_data = st.session_state.assessment_data
    
    st.markdown("### 📈 Advanced Analytics")
    
    # Performance distribution
    st.markdown("#### Score Distribution Analysis")
    
    scores = [cat_data['score'] for cat_data in assessment_data['categories'].values()]
    categories = [AssessmentCategories.CATEGORIES[cat_id]['name'] 
                 for cat_id in assessment_data['categories'].keys()]
    
    # Box plot for score distribution
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(
        y=scores,
        name="Category Scores",
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8
    ))
    
    fig_box.update_layout(
        title="Score Distribution Across Categories",
        yaxis_title="Score",
        height=400
    )
    
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Performance insights
    st.markdown("#### 🔍 Performance Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Highest scoring categories
        sorted_categories = sorted(assessment_data['categories'].items(), 
                                 key=lambda x: x[1]['score'], reverse=True)
        
        st.markdown("**🏆 Top Performing Areas:**")
        for i, (cat_id, cat_data) in enumerate(sorted_categories[:3], 1):
            category_name = AssessmentCategories.CATEGORIES[cat_id]['name']
            st.markdown(f"{i}. {category_name}: {cat_data['score']}/100")
    
    with col2:
        # Areas needing attention
        st.markdown("**⚠️ Areas Needing Attention:**")
        for i, (cat_id, cat_data) in enumerate(reversed(sorted_categories[-3:]), 1):
            category_name = AssessmentCategories.CATEGORIES[cat_id]['name']
            st.markdown(f"{i}. {category_name}: {cat_data['score']}/100")
    
    # Weighted score analysis
    st.markdown("#### ⚖️ Weighted Score Contribution")
    
    contributions = []
    for cat_id, cat_data in assessment_data['categories'].items():
        category_name = AssessmentCategories.CATEGORIES[cat_id]['name']
        weight = AssessmentCategories.CATEGORIES[cat_id]['weight']
        contribution = (cat_data['score'] * weight)
        contributions.append({
            'Category': category_name,
            'Raw Score': cat_data['score'],
            'Weight': weight * 100,
            'Weighted Contribution': contribution
        })
    
    contrib_df = pd.DataFrame(contributions)
    
    # Create stacked bar chart
    fig_contrib = px.bar(
        contrib_df,
        x='Category',
        y='Weighted Contribution',
        color='Raw Score',
        title="Category Contribution to Final Score",
        color_continuous_scale='RdYlGn'
    )
    
    fig_contrib.update_layout(height=400)
    st.plotly_chart(fig_contrib, use_container_width=True)
    
    # Recommendations
    st.markdown("#### 💡 AI Recommendations")
    
    # Generate recommendations based on scores
    recommendations = []
    
    for cat_id, cat_data in assessment_data['categories'].items():
        category_name = AssessmentCategories.CATEGORIES[cat_id]['name']
        score = cat_data['score']
        weight = AssessmentCategories.CATEGORIES[cat_id]['weight']
        
        if score < 60:
            impact = weight * (60 - score)
            recommendations.append({
                'category': category_name,
                'score': score,
                'impact': impact,
                'priority': 'High' if impact > 5 else 'Medium',
                'suggestion': f"Focus on improving {category_name.lower()} as it significantly impacts the overall assessment."
            })
        elif score < 75 and weight > 0.15:
            recommendations.append({
                'category': category_name,
                'score': score,
                'impact': weight * (75 - score),
                'priority': 'Medium',
                'suggestion': f"Enhance {category_name.lower()} skills to reach the next performance level."
            })
    
    if recommendations:
        recommendations.sort(key=lambda x: x['impact'], reverse=True)
        
        for rec in recommendations[:5]:  # Show top 5 recommendations
            priority_color = "#FEE2E2" if rec['priority'] == 'High' else "#FEF3C7"
            st.markdown(f"""
            <div class='score-card' style='background-color: {priority_color};'>
                <div class='skill-header'>{rec['priority']} Priority: {rec['category']}</div>
                <div>Current Score: {rec['score']}/100</div>
                <div style='margin-top: 0.5rem;'>{rec['suggestion']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("🎉 Great job! All categories are performing well. Focus on maintaining these standards.")

def main():
    """Main application function"""
    # Setup
    setup_page_config()
    SessionManager.initialize_session()
    
    # Header
    st.markdown("<h1 class='main-header'>AI Interview Assessment Platform</h1>", unsafe_allow_html=True)
    
    # Sidebar
    job_role, experience_level = render_sidebar()
    
    # Store job role in session for use in forms
    st.session_state.job_role = job_role
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Upload Interview", 
        "📊 Results Dashboard", 
        "🔍 Detailed Analysis",
        "📈 Analytics"
    ])
    
    # Tab 1: Upload and process interview
    with tab1:
        uploaded_file, candidate_name, position_applied = render_upload_tab()
        
        if uploaded_file and position_applied:
            with st.spinner("Processing interview..."):
                process_interview(
                    uploaded_file, candidate_name, position_applied, 
                    job_role, experience_level
                )
    
    # Tab 2: Results dashboard
    with tab2:
        render_results_dashboard()
    
    # Tab 3: Detailed analysis
    with tab3:
        render_detailed_analysis()
    
    # Tab 4: Analytics
    with tab4:
        render_analytics_tab()
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 New Analysis"):
            SessionManager.clear_session()
            st.rerun()
    
    with col2:
        st.markdown("**Session ID:** " + st.session_state.get('session_id', 'Unknown')[:8])
    
    with col3:
        st.markdown("**Version:** 2.1.0")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application crashed: {e}", exc_info=True)
        
        # Provide recovery options
        st.markdown("### 🔧 Recovery Options")
        if st.button("🔄 Restart Application"):
            st.cache_data.clear()
            st.cache_resource.clear()
            SessionManager.clear_session()
            st.rerun()
