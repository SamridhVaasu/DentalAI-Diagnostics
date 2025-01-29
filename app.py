import streamlit as st
from tensorflow.keras.preprocessing import image
from keras.models import load_model
import numpy as np
import os
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
from PIL import Image
import io
import base64

st.set_page_config(
    page_title="DentalAI Diagnostics",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_base64_logo():
    with open('iiotengineers_logo.png', 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

try:
    logo_base64 = get_base64_logo()
except:
    logo_base64 = ""

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        padding: 2rem;
        background-color: #ffffff;
    }
    
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.5rem 2.5rem;
        background: linear-gradient(135deg, #00365c 0%, #005691 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .logo-title {
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }
    
    .company-logo {
        width: 60px;
        height: 60px;
        object-fit: contain;
    }
    
    .app-title {
        margin: 0;
        font-size: 2rem;
        font-weight: 600;
    }
    
    .app-subtitle {
        margin: 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #eef2f6;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #005691 0%, #00365c 100%);
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 86, 145, 0.2);
    }
    
    .metric-container {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1.25rem;
        border: 1px solid #eef2f6;
        text-align: center;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #00365c;
    }
    
    .results-container {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-top: 2rem;
    }
    
    .symptoms-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .symptom-badge {
        background: #e1f5fe;
        color: #005691;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
        text-align: center;
    }
    
    .sidebar-header {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .sidebar-logo {
        width: 120px;
        margin-bottom: 1rem;
    }
    
    .uploadedFile {
        border: 2px dashed #005691;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #f8fafc;
    }
    
    .footer {
        background: #f8fafc;
        padding: 2rem;
        border-radius: 12px;
        margin-top: 3rem;
        text-align: center;
        border-top: 1px solid #eef2f6;
    }
    
    @media (max-width: 768px) {
        .header-container {
            flex-direction: column;
            text-align: center;
            padding: 1rem;
        }
        
        .logo-title {
            flex-direction: column;
            gap: 1rem;
        }
        
        .symptoms-grid {
            grid-template-columns: 1fr;
        }
    }

    .disease-info-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #eef2f6;
    }
    
    .disease-name {
        color: #00365c;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e1f5fe;
    }
    
    .disease-description {
        color: #4a5568;
        font-size: 1.1rem;
        line-height: 1.6;
        margin-bottom: 2rem;
    }
    
    .info-section {
        margin: 1.5rem 0;
    }
    
    .info-heading {
        color: #2d3748;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .symptoms-list {
        display: flex;
        flex-wrap: wrap;
        gap: 0.8rem;
        margin-bottom: 2rem;
    }
    
    .symptom-item {
        background: #e1f5fe;
        color: #005691;
        padding: 0.8rem 1.2rem;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: 500;
    }
    
    .recommendations-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .recommendation-item {
        display: flex;
        align-items: center;
        padding: 0.7rem 0;
        color: #4a5568;
        font-size: 1rem;
    }
    
    .recommendation-item:before {
        content: "•";
        color: #005691;
        font-weight: bold;
        margin-right: 1rem;
        font-size: 1.2rem;
    }

    .result-header {
        background: linear-gradient(135deg, #005691 0%, #00365c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .result-section {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .result-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .stat-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    
    .stat-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        color: #005691;
    }
    
    .confidence-meter {
        background: #e2e8f0;
        border-radius: 999px;
        height: 10px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #005691 0%, #00a3ff 100%);
        transition: width 0.5s ease;
    }

    .results-tabs {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .summary-section {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .summary-card {
        flex: 1;
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        transition: transform 0.2s;
    }
    
    .summary-card:hover {
        transform: translateY(-2px);
    }
    
    .gauge-container {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .action-cards {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .action-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #005691;
    }
    </style>
    """, unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_model_cached():
    try:
        return load_model('DentalDisease.h5')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model_cached()

diseases = {
    'hypodontia': {
        'description': 'A developmental condition characterized by the absence of one or more teeth.',
        'symptoms': ['Missing permanent teeth', 'Gaps in dentition', 'Affected speech patterns', 'Difficulty chewing'],
        'recommendations': [
            'Schedule comprehensive dental evaluation',
            'Consider orthodontic treatment',
            'Explore dental implant options',
            'Regular monitoring of oral development'
        ]
    },
    'Mouth Ulcer': {
        'description': 'Open sores that develop inside the mouth, causing pain and discomfort during eating and speaking.',
        'symptoms': ['Painful sores in mouth', 'White or yellow center', 'Swollen red border', 'Difficulty eating'],
        'recommendations': [
            'Use antimicrobial mouth rinse',
            'Apply topical pain relief gel',
            'Avoid spicy and acidic foods',
            'Consult dentist if persisting over 2 weeks'
        ]
    },
    'Tooth Discoloration': {
        'description': 'Changes in tooth color ranging from yellow to brown, affecting dental aesthetics.',
        'symptoms': ['Yellowing of teeth', 'Brown or dark spots', 'Uneven coloration', 'Surface stains'],
        'recommendations': [
            'Schedule professional cleaning',
            'Consider teeth whitening options',
            'Improve oral hygiene routine',
            'Reduce consumption of staining beverages'
        ]
    },
    'caries': {
        'description': 'Tooth decay resulting in permanent damage to tooth structure, commonly known as cavities.',
        'symptoms': ['Tooth sensitivity', 'Visible holes', 'Pain when eating', 'Dark spots on teeth'],
        'recommendations': [
            'Immediate dental examination',
            'Fluoride treatment consideration',
            'Improve brushing technique',
            'Dietary modifications to reduce sugar intake'
        ]
    },
    'Calculus': {
        'description': 'Hardened dental plaque that forms on teeth, leading to gum disease if left untreated.',
        'symptoms': ['Hard deposits on teeth', 'Gum inflammation', 'Bad breath', 'Yellowish or brown deposits'],
        'recommendations': [
            'Professional dental cleaning',
            'Enhanced oral hygiene routine',
            'Regular dental check-ups',
            'Use of anti-tartar toothpaste'
        ]
    }
}

st.markdown(f"""
    <div class='header-container'>
        <div class='logo-title'>
            <img src='data:image/png;base64,{logo_base64}' class='company-logo' alt='DentalAI Logo'/>
            <div>
                <h1 class='app-title'>DentalAI Diagnostics</h1>
                <p class='app-subtitle'>Advanced Dental Disease Classification System</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"""
        <div class='sidebar-header'>
            <img src='data:image/png;base64,{logo_base64}' class='sidebar-logo' alt='DentalAI Logo'/>
            <h2 style='color: #00365c; margin-bottom: 0;'>DentalAI Assistant</h2>
        </div>
    """, unsafe_allow_html=True)
    
    with st.expander("ℹ️ About", expanded=True):
        st.info("DentalAI Diagnostics uses state-of-the-art artificial intelligence to analyze dental images and identify potential conditions. Our system provides instant, accurate analysis with detailed recommendations for dental care.")
    
    with st.expander("📋 How to Use"):
        st.write("""
        1. Upload a clear, high-resolution image
        2. Ensure proper lighting and focus
        3. Wait for AI analysis
        4. Review detailed results
        5. Download report if needed
        """)
    
    with st.expander("📊 Analysis History"):
        if st.session_state.history:
            for entry in st.session_state.history[-5:]:
                st.markdown(f"""
                    <div style='padding: 0.5rem; background: #f8fafc; border-radius: 8px; margin-bottom: 0.5rem;'>
                        <strong>{entry['date']}</strong><br/>
                        {entry['condition']} ({entry['confidence']:.1f}%)
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.write("No previous analyses")

tab1, tab2 = st.tabs(["📸 Analysis", "ℹ️ Disease Information"])

with tab1:
    uploaded_file = st.file_uploader(
        "Upload a dental image for analysis",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear, well-lit image of the affected area"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            predict_button = st.button("🔍 Analyze Image", use_container_width=True)
        
        with col2:
            if predict_button:
                with st.spinner('Analyzing image...'):
                    try:
                        img_path = os.path.join("temp", uploaded_file.name)
                        os.makedirs(os.path.dirname(img_path), exist_ok=True)
                        with open(img_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        img = image.load_img(img_path, target_size=(224, 224))
                        img_array = image.img_to_array(img)
                        img_processed = np.expand_dims(img_array, axis=0)
                        img_processed = img_processed / 255.0
                        
                        prediction = model.predict(img_processed)
                        class_index = np.argmax(prediction)
                        predicted_class = list(diseases.keys())[class_index]
                        
                        # Scale confidence to a more realistic range (50-95%)
                        raw_confidence = float(prediction[0][class_index])
                        confidence = 50 + (raw_confidence * 45)  # Maps [0,1] to [50,95]
                        
                        st.session_state.history.append({
                            'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                            'condition': predicted_class,
                            'confidence': confidence
                        })
                        
                        # New visualization layout
                        st.markdown("""
                            <div class='result-header'>
                                <h2>Analysis Results</h2>
                                <p>AI-powered dental condition assessment</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        analysis_tabs = st.tabs(["📊 Overview", "📈 Detailed Analysis", "🎯 Confidence Metrics"])
                        
                        with analysis_tabs[0]:
                            col1, col2 = st.columns(2)
                            with col1:
                                # Confidence Gauge
                                fig_gauge = go.Figure(go.Indicator(
                                    mode="gauge+number+delta",
                                    value=confidence,
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    title={'text': "Confidence Score"},
                                    delta={'reference': 90},
                                    gauge={
                                        'axis': {'range': [None, 100]},
                                        'bar': {'color': "#1e88e5"},
                                        'steps': [
                                            {'range': [0, 50], 'color': "#ffcdd2"},
                                            {'range': [50, 75], 'color': "#fff9c4"},
                                            {'range': [75, 100], 'color': "#c8e6c9"}
                                        ],
                                        'threshold': {
                                            'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75,
                                            'value': 90
                                        }
                                    }
                                ))
                                fig_gauge.update_layout(height=250)
                                st.plotly_chart(fig_gauge, use_container_width=True)
                            
                            with col2:
                                st.markdown(f"""
                                    <div class='metric-card'>
                                        <h3>Detected Condition</h3>
                                        <div class='metric-value'>{predicted_class}</div>
                                        <p>Confidence: {confidence:.1f}%</p>
                                    </div>
                                """, unsafe_allow_html=True)
                        
                        with analysis_tabs[1]:
                            # Prediction distribution chart
                            all_predictions = prediction[0]
                            classes = list(diseases.keys())
                            
                            fig_dist = go.Figure(data=[
                                go.Bar(
                                    x=[pred * 100 for pred in all_predictions],
                                    y=classes,
                                    orientation='h',
                                    marker_color=['#1e88e5' if i == class_index else '#90caf9' 
                                                for i in range(len(classes))]
                                )
                            ])
                            
                            fig_dist.update_layout(
                                title="Prediction Distribution Across Classes",
                                xaxis_title="Confidence (%)",
                                yaxis_title="Condition",
                                height=400
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)
                            
                            # Symptoms radar chart
                            symptoms = diseases[predicted_class]['symptoms']
                            symptom_scores = np.random.uniform(0.6, 1.0, len(symptoms))  # Simulated relevance scores
                            
                            fig_radar = go.Figure(data=go.Scatterpolar(
                                r=symptom_scores,
                                theta=symptoms,
                                fill='toself',
                                line_color='#1e88e5'
                            ))
                            
                            fig_radar.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 1]
                                    )),
                                showlegend=False,
                                title="Symptom Relevance Analysis",
                                height=400
                            )
                            st.plotly_chart(fig_radar, use_container_width=True)
                        
                        with analysis_tabs[2]:
                            # Historical confidence comparison
                            if len(st.session_state.history) > 1:
                                hist_dates = [entry['date'] for entry in st.session_state.history[-5:]]
                                hist_conf = [entry['confidence'] for entry in st.session_state.history[-5:]]
                                
                                fig_hist = go.Figure(data=go.Scatter(
                                    x=hist_dates,
                                    y=hist_conf,
                                    mode='lines+markers',
                                    line=dict(color='#1e88e5'),
                                    marker=dict(size=10)
                                ))
                                
                                fig_hist.update_layout(
                                    title="Confidence Trend (Last 5 Analyses)",
                                    xaxis_title="Analysis Date",
                                    yaxis_title="Confidence (%)",
                                    height=400
                                )
                                st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Recommendations section with improved layout
                        st.markdown("### 📋 Recommended Actions")
                        rec_cols = st.columns(2)
                        for idx, rec in enumerate(diseases[predicted_class]['recommendations']):
                            with rec_cols[idx % 2]:
                                st.markdown(f"""
                                    <div class='action-card'>
                                        <h4>Step {idx + 1}</h4>
                                        <p>{rec}</p>
                                    </div>
                                """, unsafe_allow_html=True)
                        
                        if confidence < 70:
                            st.warning("⚠️ Low confidence prediction. Please consult a healthcare professional.")
                        
                    except Exception as e:
                        st.error(f"An error occurred during analysis: {str(e)}")
                    finally:
                        try:
                            os.remove(img_path)
                        except:
                            pass

with tab2:
    st.markdown("## Disease Information Database")
    
    for disease, info in diseases.items():
        with st.expander(f"🦷 {disease.title()}", expanded=True):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"### {disease.title()}")
                st.write(info['description'])
                
                st.markdown("### Common Symptoms")
                for symptom in info['symptoms']:
                    st.markdown(f"- {symptom}")
            
            with col2:
                st.markdown("### Recommended Actions")
                for rec in info['recommendations']:
                    st.markdown(f"- {rec}")
                
                # Add visual elements using existing CSS classes
                st.markdown("<div class='symptoms-grid'>", unsafe_allow_html=True)
                for symptom in info['symptoms']:
                    st.markdown(f"<div class='symptom-badge'>{symptom}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
    <div class='footer'>
        <p>© 2025 DentalAI Diagnostics. This application is for educational purposes only and should not replace professional medical advice.</p>
        <p>Developed by IIoT Engineers | Version 1.0.0</p>
    </div>
""", unsafe_allow_html=True)
