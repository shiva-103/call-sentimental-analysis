import streamlit as st
import os
import sys
import json
import pandas as pd
import plotly.graph_objects as go
import base64
from multi_agent_main import MultiAgentCallAnalysisSystem, AGENT_PROFILES

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Call Analysis System",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for the title section with reduced size
st.markdown("""
<style>
    .title-container {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
        padding: 0.8rem;
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
    }
    .title-icon {
        color: #e63946;
        font-size: 2.5rem;
        margin-right: 1rem;
    }
    .title-text {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #343a40 0%, #495057 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .title-subtitle {
        color: #6c757d;
        font-size: 0.9rem;
        margin-top: 0.2rem;
    }
    .highlight {
        color: #e63946;
        font-weight: 600;
    }
    .agent-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    .agent-coordinator {
        background-color: #e3f2fd;
        color: #1976d2;
    }
    .agent-analyst {
        background-color: #f3e5f5;
        color: #7b1fa2;
    }
    .agent-evaluator {
        background-color: #e8f5e8;
        color: #388e3c;
    }
    .agent-insights {
        background-color: #fff3e0;
        color: #f57c00;
    }
    .agent-qa {
        background-color: #ffebee;
        color: #d32f2f;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* Premium audio player with depth and dimension */
    .premium-audio-card {
        background: linear-gradient(135deg, #ffffff, #f5f7fa);
        border-radius: 16px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05), 0 5px 10px rgba(0,0,0,0.05);
        border: 1px solid rgba(230, 230, 230, 0.7);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .premium-audio-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 30px rgba(0,0,0,0.07), 0 7px 12px rgba(0,0,0,0.06);
    }
    
    /* Sophisticated header with improved spacing */
    .premium-audio-header {
        display: flex;
        align-items: center;
        margin-bottom: 18px;
    }
    
    /* Premium icon with subtle glow and depth */
    .premium-audio-icon {
        background: linear-gradient(45deg, #3a0ca3, #4361ee);
        color: white;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-right: 18px;
        font-size: 20px;
        box-shadow: 0 4px 12px rgba(67, 97, 238, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .premium-audio-icon::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, rgba(255,255,255,0) 70%);
        opacity: 0.6;
    }
    
    /* Elegant typography for file details */
    .premium-file-name {
        font-weight: 700;
        font-size: 17px;
        color: #2b2d42;
        margin-bottom: 6px;
        letter-spacing: 0.2px;
    }
    
    .premium-file-meta {
        font-size: 14px;
        color: #64748b;
        display: flex;
        align-items: center;
    }
    
    .premium-meta-dot {
        display: inline-block;
        width: 4px;
        height: 4px;
        border-radius: 50%;
        background-color: #cbd5e1;
        margin: 0 8px;
    }
    
    .premium-tag {
        display: inline-block;
        padding: 3px 8px;
        background-color: #eff6ff;
        color: #3b82f6;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-left: 8px;
    }
    
    /* Stylish audio player container */
    .premium-audio-container {
        background: rgba(241, 245, 249, 0.7);
        padding: 12px;
        border-radius: 12px;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(203, 213, 225, 0.5);
    }
    
    /* Custom audio element styling */
    .premium-audio-card audio {
        width: 100% !important;
        height: 44px !important;
        border-radius: 8px !important;
    }
    
    /* Custom audio controls (note: these won't affect all browsers equally) */
    .premium-audio-card audio::-webkit-media-controls-panel {
        background-color: rgba(255, 255, 255, 0.9);
    }
    
    .premium-audio-card audio::-webkit-media-controls-play-button {
        background-color: #4361ee;
        border-radius: 50%;
        transition: all 0.2s ease;
    }
    
    /* Header for audio section */
    .premium-section-header {
        font-size: 22px; 
        font-weight: 700; 
        color: #2b2d42; 
        margin: 30px 0 15px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #f1f5f9;
    }
</style>
""", unsafe_allow_html=True)

# Title with attractive styling but smaller size and multi-agent branding
st.markdown("""
<div class="title-container">
    <div class="title-icon">🤖</div>
    <div>
        <h1 class="title-text">Multi-Agent Sentiment & Response Analytics</h1>
        <p class="title-subtitle">
            AI-powered multi-agent collaboration for <span class="highlight">comprehensive call analysis</span>
            <br>
            <span class="agent-badge agent-coordinator">System Coordinator</span>
            <span class="agent-badge agent-analyst">Transcript Analyst</span>
            <span class="agent-badge agent-evaluator">Performance Evaluator</span>
            <span class="agent-badge agent-insights">Customer Insights</span>
            <span class="agent-badge agent-qa">QA Manager</span>
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'transcripts' not in st.session_state:
    st.session_state.transcripts = []
if 'call_summaries' not in st.session_state:
    st.session_state.call_summaries = []
if 'agent_category_matches' not in st.session_state:
    st.session_state.agent_category_matches = []
if 'sentiment_results' not in st.session_state:
    st.session_state.sentiment_results = []
if 'formatted_conversations' not in st.session_state:
    st.session_state.formatted_conversations = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'notification_email' not in st.session_state:
    st.session_state.notification_email = "lukkashivacharan@gmail.com"
if 'call_predictions' not in st.session_state:
    st.session_state.call_predictions = []  
if 'audio_files' not in st.session_state:
    st.session_state.audio_files = []
if 'audio_file_names' not in st.session_state:
    st.session_state.audio_file_names = [] 
if 'selected_call_index' not in st.session_state:
    st.session_state.selected_call_index = 0 

# Initialize the multi-agent call analysis system
@st.cache_resource
def get_multi_agent_system():
    return MultiAgentCallAnalysisSystem()

call_system = get_multi_agent_system()

with st.sidebar:
    st.header("⚙️ Multi-Agent Settings")
    st.session_state.notification_email = st.text_input(
        "Notification Email", 
        value=st.session_state.notification_email,
        help="Email address for alerts when human intervention is needed"
    )
    
    st.markdown("---")
    
    # Multi-Agent System Status
    st.header("🤖 Agent System Status")
    
    # Display agent status
    agents_status = [
        ("System Coordinator", "🎯", "Active", "#1976d2"),
        ("Transcript Analyst", "📋", "Active", "#7b1fa2"),
        ("Performance Evaluator", "⭐", "Active", "#388e3c"),
        ("Customer Insights", "🧠", "Active", "#f57c00"),
        ("QA Manager", "🔍", "Active", "#d32f2f")
    ]
    
    for agent_name, icon, status, color in agents_status:
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 8px; padding: 8px; background-color: #f8f9fa; border-radius: 5px;">
            <div style="font-size: 20px; margin-right: 10px;">{icon}</div>
            <div style="flex-grow: 1;">
                <div style="font-weight: bold; font-size: 14px;">{agent_name}</div>
                <div style="color: {color}; font-size: 12px;">● {status}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Agent Information Section
    st.header("👥 Available CS Agents")
    
    # Loop through all agents in AGENT_PROFILES
    for agent_name, profile in AGENT_PROFILES.items():
        with st.expander(f"{agent_name} ({profile['expertise_level']})"):
            st.markdown(f"**ID:** {profile['id']}")
            st.markdown(f"**Department:** {profile['department']}")
            st.markdown(f"**Expertise Level:** {profile['expertise_level']}")
            
            # Display categories/specialties
            st.markdown("**Specialties:**")
            for category in profile['categories']:
                st.markdown(f"- {category}")

# Main interface for file upload
uploaded_files = st.file_uploader(
    "Upload Audio Files (.wav, .mp3, .m4a)", 
    type=["wav", "mp3", "m4a"], 
    accept_multiple_files=True
)

def create_compact_audio_player(file_bytes, file_name):
    """Create a compact, minimalist audio player that takes minimal screen space"""
    try:
        file_size_mb = round(len(file_bytes) / (1024 * 1024), 2)
    except:
        file_size_mb = "Unknown"
    
    file_extension = file_name.split('.')[-1].lower()
    
    player_html = f"""
    <div style="background: #f8f9fa; border-radius: 8px; padding: 10px; margin: 10px 0; border: 1px solid #e9ecef;">
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <div style="color: #4361ee; margin-right: 8px; font-size: 18px;">🎵</div>
            <div style="font-size: 14px; font-weight: 500; color: #333;">{file_name}</div>
            <div style="margin-left: auto; font-size: 12px; color: #6c757d;">{file_size_mb} MB</div>
        </div>
    </div>
    """
    
    st.markdown(player_html, unsafe_allow_html=True)
    st.audio(file_bytes, format=f"audio/{file_extension}")

if uploaded_files:
    st.markdown("<p style='font-size: 16px; font-weight: 500; color: #333; margin-bottom: 10px;'>📁 Audio Preview</p>", 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 5])
    
    with col1:
        file_names = [uploaded_file.name for uploaded_file in uploaded_files]
        
        selected_index = st.selectbox(
            "Select audio file",
            range(len(file_names)),
            format_func=lambda i: file_names[i],
            label_visibility="collapsed"
        )
    
    with col2:
        selected_file = uploaded_files[selected_index]
        st.audio(selected_file)

# Process uploaded files with multi-agent analysis
if uploaded_files:
    if st.button("🚀 Start Multi-Agent Analysis", type="primary"):
        with st.spinner("🤖 Multi-Agent System Processing..."):
            # Save uploaded files temporarily
            temp_file_paths = []
            st.session_state.audio_files = []
            st.session_state.audio_file_names = []
            
            for uploaded_file in uploaded_files:
                temp_file_path = f"temp_{uploaded_file.name}"
                
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                file_bytes = uploaded_file.getvalue()
                st.session_state.audio_files.append(file_bytes)
                st.session_state.audio_file_names.append(uploaded_file.name)
                
                temp_file_paths.append(temp_file_path)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Upload and transcribe audio files
            status_text.text("🎤 Transcribing audio files...")
            progress_bar.progress(20)
            call_system.upload_audio_files(temp_file_paths)
            
            # Step 2: Get text transcripts
            status_text.text("📝 Processing transcripts...")
            progress_bar.progress(40)
            st.session_state.transcripts = call_system.get_text_transcripts()
            
            if not st.session_state.transcripts:
                st.error("❌ No transcripts generated. Please check the audio files.")
            else:
                # Step 3: Perform sentiment analysis
                status_text.text("😊 Analyzing sentiment...")
                progress_bar.progress(60)
                st.session_state.sentiment_results = call_system.perform_sentiment_analysis()
                
                # Format sentiment data for visualization
                if st.session_state.sentiment_results:
                    st.session_state.formatted_conversations = call_system.format_sentiment_data(
                        st.session_state.sentiment_results
                    )
                
                # Step 4: Multi-Agent Analysis
                status_text.text("🤖 Multi-Agent Collaborative Analysis...")
                progress_bar.progress(80)
                
                st.session_state.call_summaries = []
                st.session_state.agent_category_matches = []
                
                for i in range(len(st.session_state.transcripts)):
                    # Use multi-agent analysis
                    agent_match = call_system.multi_agent_analysis(i)
                    st.session_state.agent_category_matches.append(agent_match)
                    
                    # Extract call summary from multi-agent result
                    if agent_match and agent_match.get('success', False):
                        call_summary = agent_match.get('call_summary', {})
                        st.session_state.call_summaries.append(call_summary)
                    else:
                        st.session_state.call_summaries.append({})
                
                status_text.text("🧹 Cleaning up...")
                progress_bar.progress(100)
                
                # Clean up temporary files
                for temp_file in temp_file_paths:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                
                st.session_state.processing_complete = True
                status_text.text("✅ Multi-Agent Analysis Complete!")
                st.success("🎉 Multi-Agent processing complete! Check the analysis tabs below.")

if st.session_state.audio_file_names:
    call_options = st.session_state.audio_file_names
elif st.session_state.call_summaries:
    call_options = [f"Call {i+1}" for i in range(len(st.session_state.call_summaries))]
else:
    call_options = []

if call_options:
    selected_call_index = st.selectbox(
        "📞 Select Call for Analysis", 
        range(len(call_options)), 
        format_func=lambda x: call_options[x],
        key="global_call_selector",
        index=st.session_state.selected_call_index
    )
    
    st.session_state.selected_call_index = selected_call_index

# Display results if processing is complete
if st.session_state.processing_complete:
    tabs = st.tabs(["📋 Call Summary", "🤖 Multi-Agent Analysis", "😊 Sentiment Analysis"])
    
    # Call Summary Tab (Tab 0)
    with tabs[0]:
        st.header("📋 Call Summary & Overview")
        if st.session_state.call_summaries and call_options:
            selected_summary = st.session_state.call_summaries[st.session_state.selected_call_index]
            
            # Show multi-agent insights if available
            if st.session_state.agent_category_matches:
                agent_match = st.session_state.agent_category_matches[st.session_state.selected_call_index]
                if agent_match and agent_match.get('success', False):
                    qa_decision = agent_match.get('qa_decision', {})
                    customer_insights = agent_match.get('customer_insights', {})
                    
                    # Multi-agent insights banner
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                        <h4 style="margin: 0; color: white;">🤖 Multi-Agent Analysis Insights</h4>
                        <div style="margin-top: 10px; display: flex; justify-content: space-between; flex-wrap: wrap;">
                            <div style="background: rgba(255,255,255,0.2); padding: 8px 12px; border-radius: 8px; margin: 2px;">
                                <strong>QA Decision:</strong> {qa_decision.get('intervention_type', 'N/A')}
                            </div>
                            <div style="background: rgba(255,255,255,0.2); padding: 8px 12px; border-radius: 8px; margin: 2px;">
                                <strong>Priority:</strong> {qa_decision.get('priority_level', 'N/A')}
                            </div>
                            <div style="background: rgba(255,255,255,0.2); padding: 8px 12px; border-radius: 8px; margin: 2px;">
                                <strong>Satisfaction:</strong> {customer_insights.get('customer_satisfaction_prediction', 'N/A')}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Create two columns
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Overview section with improved styling
                st.markdown("### 📊 Overview")
                
                # Topic with icon
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="background-color: #f0f2f6; border-radius: 50%; width: 32px; height: 32px; display: flex; justify-content: center; align-items: center; margin-right: 10px;">
                        <span style="font-size: 16px;">📋</span>
                    </div>
                    <div>
                        <div style="font-size: 14px; color: #555;">Topic</div>
                        <div style="font-weight: bold;">{selected_summary.get('Topic', 'N/A')}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Product with icon
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="background-color: #f0f2f6; border-radius: 50%; width: 32px; height: 32px; display: flex; justify-content: center; align-items: center; margin-right: 10px;">
                        <span style="font-size: 16px;">🏷️</span>
                    </div>
                    <div>
                        <div style="font-size: 14px; color: #555;">Product</div>
                        <div style="font-weight: bold;">{selected_summary.get('Product', 'N/A')}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Summary section
                st.markdown("#### 📝 Summary")
                st.markdown(f'<div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #4b7bec;">{selected_summary.get("Summary", "N/A")}</div>', unsafe_allow_html=True)
                
                # Action Taken section with improved styling
                st.markdown("### 🎯 Action Taken")
                
                action_data = selected_summary.get('Action', 'No actions recorded')

                # Handle both string and list formats for action
                if isinstance(action_data, list):
                    action_items = action_data
                elif isinstance(action_data, str):
                    import re
                    action_items = re.split(r'[,.;]\s*', action_data)
                    action_items = [item for item in action_items if item.strip()]
                else:
                    action_items = ["No actions recorded"]

                if len(action_items) <= 1 and action_items[0] == "No actions recorded":
                    st.markdown(f'<div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db;">No actions recorded</div>', unsafe_allow_html=True)
                else:
                    for i, action in enumerate(action_items):
                        if not action or not isinstance(action, str) or not action.strip():
                            continue
                        st.markdown(f"""
                        <div style="display: flex; margin-bottom: 10px;">
                            <div style="background-color: #3498db; color: white; border-radius: 50%; width: 24px; height: 24px; display: flex; justify-content: center; align-items: center; margin-right: 10px;">
                                <span style="font-size: 12px;">{i+1}</span>
                            </div>
                            <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; flex-grow: 1;">
                                {action.strip()}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Enhanced alert for human intervention using multi-agent decision
                if st.session_state.agent_category_matches:
                    agent_match = st.session_state.agent_category_matches[st.session_state.selected_call_index]
                    if agent_match and agent_match.get('success', False):
                        qa_decision = agent_match.get('qa_decision', {})
                        intervention_type = qa_decision.get('intervention_type', 'none')
                        reasoning = qa_decision.get('reasoning', '')
                        
                        if intervention_type != "none":
                            if intervention_type == "urgent_email":
                                message = f"🚨 **URGENT**: Multi-agent system detected urgent intervention needed. {reasoning}"
                                st.error(message)
                            elif intervention_type == "high_priority_ticket":
                                message = f"⚠️ **HIGH PRIORITY**: Multi-agent analysis flagged for priority handling. {reasoning}"
                                st.warning(message)
                            elif intervention_type == "normal_ticket":
                                message = f"📋 **STANDARD**: Ticket created based on multi-agent analysis. {reasoning}"
                                st.info(message)
                            
                            # Show recommended actions
                            recommended_actions = qa_decision.get('recommended_actions', [])
                            if recommended_actions:
                                # Ensure recommended_actions is a list and properly formatted
                                if isinstance(recommended_actions, str):
                                    # Try to parse as JSON first
                                    try:
                                        import json
                                        recommended_actions = json.loads(recommended_actions)
                                    except:
                                        # Split by common delimiters if JSON parsing fails
                                        import re
                                        if any(delimiter in recommended_actions for delimiter in ['. ', ', ', '; ', '\n']):
                                            # Split by multiple delimiters
                                            recommended_actions = re.split(r'[.,;\n]\s*', recommended_actions)
                                            recommended_actions = [action.strip() for action in recommended_actions if action.strip()]
                                        else:
                                            recommended_actions = [recommended_actions]
                                
                                if isinstance(recommended_actions, list) and recommended_actions:
                                    st.markdown("**🎯 QA Recommended Actions:**")
                                    for i, action in enumerate(recommended_actions):
                                        if action and isinstance(action, str) and action.strip():
                                            st.markdown(f"• {action.strip()}")
                                        elif action:
                                            st.markdown(f"• {str(action).strip()}")
                
            with col2:
                # Call Details Card with improved styling
                st.markdown("### 📊 Call Details")
                
                # Convert values to display with icons
                details_list = []
                
                # 1. Resolved
                resolved_value = selected_summary.get('Resolved', 'N/A')
                if resolved_value == "Yes" or resolved_value == "True" or resolved_value == True:
                    resolved_icon = "✅"
                    resolved_color = "green"
                elif resolved_value == "No" or resolved_value == "False" or resolved_value == False:
                    resolved_icon = "❌"
                    resolved_color = "red"
                else:
                    resolved_icon = "⚪"
                    resolved_color = "gray"
                details_list.append(("Resolved", resolved_value, resolved_icon, resolved_color))
                
                # 2. Callback Required
                callback_value = selected_summary.get('Callback', 'N/A')
                if callback_value == "Yes" or callback_value == "True" or callback_value == True:
                    callback_icon = "🔴"  
                    callback_color = "#dc3545"  
                elif callback_value == "No" or callback_value == "False" or callback_value == False:
                    callback_icon = "🟢"  
                    callback_color = "#28a745" 
                else:
                    callback_icon = "⭕"  
                    callback_color = "#495057"  
                details_list.append(("Callback Required", callback_value, callback_icon, callback_color))
                
                # 3. Politeness
                politeness_value = selected_summary.get('Politeness', 'N/A')
                if politeness_value == "High" or politeness_value == "Excellent":
                    politeness_icon = "😊"
                    politeness_color = "green"
                elif politeness_value == "Medium" or politeness_value == "Average":
                    politeness_icon = "😐"
                    politeness_color = "orange"
                elif politeness_value == "Low" or politeness_value == "Poor":
                    politeness_icon = "😟"
                    politeness_color = "red"
                else:
                    politeness_icon = "⚪"
                    politeness_color = "gray"
                details_list.append(("Politeness", politeness_value, politeness_icon, politeness_color))
                
                # 4. Customer Sentiment
                cust_sentiment_value = selected_summary.get('Customer sentiment', 'N/A')
                if cust_sentiment_value == "POSITIVE" or cust_sentiment_value == "Positive":
                    cust_sentiment_icon = "😊"
                    cust_sentiment_color = "green"
                elif cust_sentiment_value == "NEGATIVE" or cust_sentiment_value == "Negative":
                    cust_sentiment_icon = "😠"
                    cust_sentiment_color = "red"
                elif cust_sentiment_value == "NEUTRAL" or cust_sentiment_value == "Neutral":
                    cust_sentiment_icon = "😐"
                    cust_sentiment_color = "gray"
                else:
                    cust_sentiment_icon = "⚪"
                    cust_sentiment_color = "gray"
                details_list.append(("Customer Sentiment", cust_sentiment_value, cust_sentiment_icon, cust_sentiment_color))
                
                # 5. Agent Sentiment
                agent_sentiment_value = selected_summary.get('Agent sentiment', 'N/A')
                if agent_sentiment_value == "POSITIVE" or agent_sentiment_value == "Positive":
                    agent_sentiment_icon = "😊"
                    agent_sentiment_color = "green"
                elif agent_sentiment_value == "NEGATIVE" or agent_sentiment_value == "Negative":
                    agent_sentiment_icon = "😠"
                    agent_sentiment_color = "red"
                elif agent_sentiment_value == "NEUTRAL" or agent_sentiment_value == "Neutral":
                    agent_sentiment_icon = "😐"
                    agent_sentiment_color = "gray"
                else:
                    agent_sentiment_icon = "⚪"
                    agent_sentiment_color = "gray"
                details_list.append(("Agent Sentiment", agent_sentiment_value, agent_sentiment_icon, agent_sentiment_color))
                
                # Display the details as a styled list
                for metric, value, icon, color in details_list:
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; padding: 10px; border-bottom: 1px solid #eee; margin-bottom: 5px;">
                        <div style="font-size: 24px; margin-right: 15px; color: {color};">{icon}</div>
                        <div style="flex-grow: 1;">
                            <div style="font-size: 14px; color: #555;">{metric}</div>
                            <div style="font-weight: bold; color: {color};">{value}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Multi-Agent Insights section
                
                
                # Additional metrics or insights section
                st.markdown("### 🚀 Quick Insights")
                
                # Check for flags that might need attention
                flags = []
                
                if cust_sentiment_value in ["NEGATIVE", "Negative"]:
                    flags.append(("Customer is upset", "The customer expressed negative sentiment during the call."))
                
                if resolved_value in ["No", "False", False]:
                    flags.append(("Unresolved issue", "This call did not resolve the customer's issue."))
                    
                if callback_value in ["Yes", "True", True]:
                    flags.append(("Requires follow-up", "This call requires a callback to the customer."))
                
                # Display flags if any
                if flags:
                    for flag_title, flag_desc in flags:
                        st.markdown(f"""
                        <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #ffc107;">
                            <div style="font-weight: bold;">⚠️ {flag_title}</div>
                            <div style="font-size: 14px;">{flag_desc}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #d1e7dd; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #198754;">
                        <div style="font-weight: bold;">✅ All good</div>
                        <div style="font-size: 14px;">No immediate issues detected with this call.</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No call summaries available.")
    
    # Multi-Agent Analysis Tab (Tab 1) 
    with tabs[1]:
        st.header("🤖 Multi-Agent Performance Analysis")
        if st.session_state.agent_category_matches and call_options:
        
            agent_match = st.session_state.agent_category_matches[st.session_state.selected_call_index]
            
            # Create columns for better layout
            col1, col2 = st.columns([3, 2])
            
            with col1:
                if agent_match and agent_match.get('success', False):
                    # Multi-Agent Analysis Overview
                    st.markdown("### 🤖 Multi-Agent Collaboration Results")
                    
                    # Show which agents participated
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                        <h5 style="margin: 0; color: white;">👥 Agent Collaboration Workflow</h5>
                        <div style="margin-top: 10px;">
                            <div style="display: flex; align-items: center; margin: 5px 0;">
                                <span style="background: rgba(255,255,255,0.3); border-radius: 50%; width: 20px; height: 20px; display: flex; justify-content: center; align-items: center; margin-right: 8px; font-size: 12px;">1</span>
                                <span>Transcript Analyst identified agent and categorized call</span>
                            </div>
                            <div style="display: flex; align-items: center; margin: 5px 0;">
                                <span style="background: rgba(255,255,255,0.3); border-radius: 50%; width: 20px; height: 20px; display: flex; justify-content: center; align-items: center; margin-right: 8px; font-size: 12px;">2</span>
                                <span>Performance Evaluator assessed agent compliance</span>
                            </div>
                            <div style="display: flex; align-items: center; margin: 5px 0;">
                                <span style="background: rgba(255,255,255,0.3); border-radius: 50%; width: 20px; height: 20px; display: flex; justify-content: center; align-items: center; margin-right: 8px; font-size: 12px;">3</span>
                                <span>Customer Insights analyzed needs and satisfaction</span>
                            </div>
                            <div style="display: flex; align-items: center; margin: 5px 0;">
                                <span style="background: rgba(255,255,255,0.3); border-radius: 50%; width: 20px; height: 20px; display: flex; justify-content: center; align-items: center; margin-right: 8px; font-size: 12px;">4</span>
                                <span>QA Manager determined intervention requirements</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.subheader("👤 Agent Information")
                    
                    # Create agent profile card 
                    agent_profile = agent_match.get('agent_profile', {})
                    agent_name = agent_match.get('agent_name', 'Unknown')
                    category = agent_match.get('category', 'Unknown')
                    is_authorized = agent_match.get('is_authorized', False)
                    
                    # Agent identification box with styling
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                        <div style="display: flex; align-items: center;">
                            <div style="background-color: #e9ecef; border-radius: 50%; width: 60px; height: 60px; display: flex; justify-content: center; align-items: center; margin-right: 15px;">
                                <span style="font-size: 30px;">👤</span>
                            </div>
                            <div>
                                <h3 style="margin: 0;">{agent_name}</h3>
                                <div style="color: #6c757d; font-size: 14px;">{agent_profile.get('department', 'Unknown Department')} • {agent_profile.get('expertise_level', 'Unknown')} Level</div>
                                <div style="margin-top: 5px;">
                                    <span style="background-color: {'#d1e7dd' if is_authorized else '#f8d7da'}; color: {'#0f5132' if is_authorized else '#842029'}; padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: bold;">
                                        {'✓ Authorized' if is_authorized else '✗ Not Authorized'} for {category}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Resolution Quality
                    if 'evaluation' in agent_match and 'resolution_quality' in agent_match['evaluation']:
                        st.subheader("🎯 Resolution Quality")
                        resolution = agent_match['evaluation']['resolution_quality']
                        
                        # Parse resolution string for "Yes/No/Partial" status
                        import re
                        resolution_status = "Unknown"
                        
                        # Try to extract resolution status
                        status_match = re.search(r'(Yes|No|Partial)', str(resolution))
                        if status_match:
                            resolution_status = status_match.group(1)
                        
                        # Determine color based on status
                        status_color = "#28a745" if resolution_status == "Yes" else "#dc3545" if resolution_status == "No" else "#ffc107"
                        
                        # Display only the Issue Resolved part
                        st.markdown(f"""
                        <div style="display: flex; margin-bottom: 20px;">
                            <div style="flex: 1; background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid {status_color};">
                                <div style="font-size: 14px; color: #555;">Issue Resolved</div>
                                <div style="font-weight: bold; color: {status_color};">{resolution_status}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    if 'evaluation' in agent_match:
                        # Get pain points and next actions from multi-agent analysis
                        pain_points = agent_match['evaluation'].get('customer_pain_points', [])
                        next_actions = agent_match['evaluation'].get('next_best_actions', [])
                        
                        # Also get customer insights
                        customer_insights = agent_match.get('customer_insights', {})
                        underlying_needs = customer_insights.get('underlying_needs', [])
                        recommended_actions = customer_insights.get('next_best_actions', [])
                        
                        # Process pain points to ensure it's a list
                        if isinstance(pain_points, str):
                            if '[' in pain_points and ']' in pain_points:
                                try:
                                    import ast
                                    pain_points = ast.literal_eval(pain_points)
                                except:
                                    if ',' in pain_points:
                                        pain_points = [p.strip() for p in pain_points.split(',') if p.strip()]
                                    elif '.' in pain_points:
                                        pain_points = [p.strip() for p in pain_points.split('.') if p.strip()]
                                    else:
                                        pain_points = [pain_points]
                            else:
                                if ',' in pain_points:
                                    pain_points = [p.strip() for p in pain_points.split(',') if p.strip()]
                                elif '.' in pain_points:
                                    pain_points = [p.strip() for p in pain_points.split('.') if p.strip()]
                                else:
                                    pain_points = [pain_points]
                        
                        # Combine pain points with underlying needs - ensure both are lists
                        if not isinstance(pain_points, list):
                            pain_points = [pain_points] if pain_points else []
                        if not isinstance(underlying_needs, list):
                            underlying_needs = [underlying_needs] if underlying_needs else []
                        all_issues = pain_points + underlying_needs
                        
                        # Process next actions to ensure it's a list
                        if isinstance(next_actions, str):
                            if '[' in next_actions and ']' in next_actions:
                                try:
                                    import ast
                                    next_actions = ast.literal_eval(next_actions)
                                except:
                                    import re
                                    if re.search(r'\d+\.', next_actions):
                                        next_actions = re.split(r'\d+\.', next_actions)
                                        next_actions = [a.strip() for a in next_actions if a.strip()]
                                    elif ',' in next_actions:
                                        next_actions = [a.strip() for a in next_actions.split(',') if a.strip()]
                                    elif '.' in next_actions:
                                        next_actions = [a.strip() for a in next_actions.split('.') if a.strip()]
                                    else:
                                        next_actions = [next_actions]
                            else:
                                import re
                                if re.search(r'\d+\.', next_actions):
                                    next_actions = re.split(r'\d+\.', next_actions)
                                    next_actions = [a.strip() for a in next_actions if a.strip()]
                                elif ',' in next_actions:
                                    next_actions = [a.strip() for a in next_actions.split(',') if a.strip()]
                                elif '.' in next_actions:
                                    next_actions = [a.strip() for a in next_actions.split('.') if a.strip()]
                                else:
                                    next_actions = [next_actions]
                        
                        # Combine actions from different agents - ensure both are lists
                        if not isinstance(next_actions, list):
                            next_actions = [next_actions] if next_actions else []
                        if not isinstance(recommended_actions, list):
                            recommended_actions = [recommended_actions] if recommended_actions else []
                        all_actions = next_actions + recommended_actions
                        
                        # Add CSS for the cards
                        st.markdown("""
                        <style>
                            .issue-card {
                                background-color: white;
                                border-radius: 6px;
                                padding: 12px;
                                margin-bottom: 10px;
                                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                                border-top: 4px solid #e74c3c;
                                transition: transform 0.2s ease, box-shadow 0.2s ease;
                            }
                            
                            .issue-card:hover {
                                transform: translateY(-2px);
                                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                            }
                            
                            .action-card {
                                background-color: white;
                                border-radius: 6px;
                                padding: 12px;
                                margin-bottom: 10px;
                                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                                border-top: 4px solid #3498db;
                                transition: transform 0.2s ease, box-shadow 0.2s ease;
                            }
                            
                            .action-card:hover {
                                transform: translateY(-2px);
                                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                            }
                            
                            .badge {
                                display: inline-block;
                                padding: 2px 6px;
                                border-radius: 12px;
                                font-size: 11px;
                                font-weight: 600;
                                margin-bottom: 5px;
                            }
                            
                            .badge-action {
                                background-color: #e3f2fd;
                                color: #2196f3;
                            }
                            
                            .badge-issue {
                                background-color: #ffebee;
                                color: #f44336;
                            }
                            
                            .card-content {
                                font-size: 13px;
                                color: #444;
                                line-height: 1.4;
                            }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        # First section: Customer Issues & Needs
                        st.subheader("🎯 Customer Issues & Needs")
                        
                        # Create columns for issues (2 columns side by side)
                        issue_cols = st.columns(2)
                        
                        # Distribute issues across columns
                        for i, issue in enumerate(all_issues):
                            if not issue or not isinstance(issue, str) or not issue.strip():
                                continue
                            
                            col_idx = i % 2
                            
                            with issue_cols[col_idx]:
                                st.markdown(f"""
                                <div class="issue-card">
                                    <div class="badge badge-issue">INSIGHT</div>
                                    <div class="card-content">Issue #{i+1}: {issue.strip()}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Add space between sections
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Second section: Multi-Agent Recommended Actions
                        st.subheader("🚀 Multi-Agent Recommended Actions")
                        
                        # Create columns for actions (2 columns side by side)
                        action_cols = st.columns(2)
                        
                        # Distribute actions across columns
                        for i, action in enumerate(all_actions):
                            if not action or not isinstance(action, str) or not action.strip():
                                continue
                            
                            col_idx = i % 2
                            
                            with action_cols[col_idx]:
                                st.markdown(f"""
                                <div class="action-card">
                                    <div class="badge badge-action">AI RECOMMENDATION</div>
                                    <div class="card-content">Action #{i+1}: {action.strip()}</div>
                                </div>
                                """, unsafe_allow_html=True)

                    # Display company standards compliance
                    st.markdown("#### 📋 Standards Compliance")
                    eval_data = agent_match['evaluation']
                    if 'standards_met' in eval_data:
                        standards = eval_data['standards_met']
                        
                        # Calculate compliance rate
                        standards_count = len(standards)
                        yes_count = sum(1 for val in standards.values() if val == "Yes")
                        partial_count = sum(1 for val in standards.values() if val == "Partial")
                        no_count = sum(1 for val in standards.values() if val == "No")
                        
                        compliance_rate = (yes_count + (partial_count * 0.5)) / standards_count * 100
                        
                        # Display compliance summary
                        st.markdown(f"""
                        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                            <h5 style="margin-top: 0;">Compliance Rate: {compliance_rate:.1f}%</h5>
                            <div style="display: flex; height: 20px; width: 100%; border-radius: 10px; overflow: hidden; margin-bottom: 10px;">
                                <div style="width: {yes_count/standards_count*100}%; background-color: #28a745;"></div>
                                <div style="width: {partial_count/standards_count*100}%; background-color: #ffc107;"></div>
                                <div style="width: {no_count/standards_count*100}%; background-color: #dc3545;"></div>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 12px; max-width: 300px;">
                                <div><span style="color: #28a745;">■</span> Yes ({yes_count})</div>
                                <div><span style="color: #ffc107;">■</span> Partial ({partial_count})</div>
                                <div><span style="color: #dc3545;">■</span> No ({no_count})</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Use a collapsible section for all standards
                        with st.expander("📋 View All Standards Details"):
                            # Sort standards by compliance status (No, Partial, Yes)
                            sorted_standards = sorted(
                                standards.items(), 
                                key=lambda x: 0 if x[1] == "No" else 1 if x[1] == "Partial" else 2
                            )
                            
                            for standard, compliance in sorted_standards:
                                # Set color based on compliance
                                color = "#28a745" if compliance == "Yes" else "#dc3545" if compliance == "No" else "#ffc107"
                                icon = "✓" if compliance == "Yes" else "✗" if compliance == "No" else "⚠"
                                
                                st.markdown(f"""
                                <div style="display: flex; align-items: center; margin-bottom: 8px; background-color: #f8f9fa; padding: 8px; border-radius: 5px; border-left: 4px solid {color};">
                                    <div style="background-color: {color}; color: white; border-radius: 50%; width: 24px; height: 24px; display: flex; justify-content: center; align-items: center; margin-right: 10px;">
                                        <span>{icon}</span>
                                    </div>
                                    <div style="flex-grow: 1;">
                                        <div style="font-size: 14px;">{standard}</div>
                                    </div>
                                    <div style="font-weight: bold; color: {color}; margin-left: 10px;">{compliance}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        # Replace the simple info message with more specific error handling
                        if not agent_match:
                            st.error("📋 No agent analysis data available for this call")
                        elif not agent_match.get('success', False):
                            st.warning("⚠️ Multi-agent system could not complete analysis")
                            if 'error' in agent_match:
                                st.info(f"Error: {agent_match['error']}")
                        elif 'evaluation' not in agent_match:
                            st.info(f"📊 Agent identified as {agent_match.get('agent_name', 'Unknown')}, but evaluation data is incomplete")
                        else:
                            agent_name = agent_match.get('agent_name', 'Unknown')
                            if agent_name.lower() == "unidentified" or agent_name.lower() == "unknown":
                                st.info("⚠️ Multi-agent system could not identify the agent for this call")
                            else:
                                st.info("No evaluation data available for this agent")

            with col2:
                if agent_match and agent_match.get('success', False) and 'evaluation' in agent_match:
                    eval_data = agent_match['evaluation']
                    
                    # Overall Rating with gauge
                    st.subheader("⭐ Performance Rating")
                    
                    # Display overall rating with gauge visualization
                    if 'overall_rating' in eval_data:
                        try:
                            rating_value = float(eval_data['overall_rating'])
                            
                            # Determine color based on rating
                            color = "#dc3545" if rating_value < 4 else "#ffc107" if rating_value < 7 else "#28a745"
                            
                            # Create gauge chart using plotly with improved number display
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = rating_value,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Multi-Agent Rating", 'font': {'size': 24}},
                                number = {'font': {'size': 40, 'color': color}},
                                gauge = {
                                    'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                    'bar': {'color': color},
                                    'bgcolor': "white",
                                    'borderwidth': 2,
                                    'bordercolor': "gray",
                                    'steps': [
                                        {'range': [0, 4], 'color': 'rgba(220, 53, 69, 0.2)'},
                                        {'range': [4, 7], 'color': 'rgba(255, 193, 7, 0.2)'},
                                        {'range': [7, 10], 'color': 'rgba(40, 167, 69, 0.2)'}
                                    ],
                                }
                            ))
                            
                            fig.update_layout(
                                height=300,
                                margin=dict(l=10, r=10, t=60, b=20),
                                paper_bgcolor="white",
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        except (ValueError, TypeError):
                            st.markdown(f"**Overall Rating:** {eval_data.get('overall_rating', 'N/A')}")
                    
                    # Category Expertise
                    if 'category_expertise' in eval_data:
                        category_expertise = eval_data['category_expertise']
                        expertise_color = "#28a745" if category_expertise == "High" else "#ffc107" if category_expertise == "Medium" else "#dc3545"
                        
                        st.markdown(f"""
                        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid {expertise_color};">
                            <div style="font-size: 14px; color: #555;">Category Expertise</div>
                            <div style="font-weight: bold; color: {expertise_color};">{category_expertise}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Multi-Agent Customer Insights
                    customer_insights = agent_match.get('customer_insights', {})
                    if customer_insights:
                        st.markdown("### 🧠 AI Customer Insights")
                        
                        # Customer Satisfaction Prediction
                        satisfaction = customer_insights.get('customer_satisfaction_prediction', 'Unknown')
                        satisfaction_color = "#28a745" if satisfaction == "High" else "#ffc107" if satisfaction == "Medium" else "#dc3545"
                        
                        st.markdown(f"""
                        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid {satisfaction_color};">
                            <div style="font-size: 14px; color: #555;">Predicted Satisfaction</div>
                            <div style="font-weight: bold; color: {satisfaction_color};">{satisfaction}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Follow-up Priority
                        priority = customer_insights.get('follow_up_priority', 'Unknown')
                        priority_color = "#dc3545" if priority == "High" else "#ffc107" if priority == "Medium" else "#28a745"
                        
                        st.markdown(f"""
                        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid {priority_color};">
                            <div style="font-size: 14px; color: #555;">Follow-up Priority</div>
                            <div style="font-weight: bold; color: {priority_color};">{priority}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Empathy Rating
                    if 'agent_empathy' in eval_data:
                        try:
                            empathy_score = float(eval_data['agent_empathy'])
                            empathy_color = "#dc3545" if empathy_score < 4 else "#ffc107" if empathy_score < 7 else "#28a745"
                            
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid {empathy_color};">
                                <div style="font-size: 14px; color: #555;">Empathy Rating</div>
                                <div style="font-weight: bold; color: {empathy_color};">{empathy_score}/10</div>
                            </div>
                            """, unsafe_allow_html=True)
                        except (ValueError, TypeError):
                            st.markdown(f"**Empathy Rating:** {eval_data.get('agent_empathy', 'N/A')}")

                    if 'would_recommend' in eval_data:
                        recommend = eval_data['would_recommend']
                        recommend_color = "#28a745" if recommend == "Yes" else "#dc3545" if recommend == "No" else "#ffc107"
                        recommend_icon = "👍" if recommend == "Yes" else "👎" if recommend == "No" else "🤔"
                        
                        st.markdown(f"""
                        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid {recommend_color};">
                            <div style="display: flex; align-items: center;">
                                <div style="font-size: 24px; margin-right: 15px;">{recommend_icon}</div>
                                <div>
                                    <div style="font-size: 14px; color: #555;">Would Recommend Agent</div>
                                    <div style="font-weight: bold; color: {recommend_color};">{recommend}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # QA Manager Decision
                    qa_decision = agent_match.get('qa_decision', {})
                    if qa_decision:
                        st.markdown("### 🔍 QA Decision")
                        
                        intervention_type = qa_decision.get('intervention_type', 'none')
                        qa_color = "#dc3545" if intervention_type == "urgent_email" else "#ffc107" if intervention_type == "high_priority_ticket" else "#28a745"
                        
                        st.markdown(f"""
                        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid {qa_color};">
                            <div style="font-size: 14px; color: #555;">Intervention Level</div>
                            <div style="font-weight: bold; color: {qa_color};">{intervention_type.replace('_', ' ').title()}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # QA Reasoning
                        reasoning = qa_decision.get('reasoning', '')
                        if reasoning:
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 12px; border-radius: 5px; margin-bottom: 15px; font-size: 13px;">
                                <strong>QA Reasoning:</strong> {reasoning}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Strengths and Areas for Improvement
                    strengths = eval_data.get('strengths', [])
                    improvements = eval_data.get('areas_for_improvement', [])
                    
                    if strengths or improvements:
                        st.markdown("### 💪 Performance Summary")
                        
                        if strengths:
                            st.markdown("**✅ Strengths:**")
                            for strength in strengths[:3]:  # Limit to top 3
                                st.markdown(f"""
                                <div style="background-color: #d1e7dd; border-left: 4px solid #28a745; padding: 8px; margin-bottom: 6px; border-radius: 4px; font-size: 13px;">
                                    {strength}
                                </div>
                                """, unsafe_allow_html=True)
                        
                        if improvements:
                            st.markdown("**⚠️ Areas for Improvement:**")
                            for improvement in improvements[:3]:  # Limit to top 3
                                st.markdown(f"""
                                <div style="background-color: #f8d7da; border-left: 4px solid #dc3545; padding: 8px; margin-bottom: 6px; border-radius: 4px; font-size: 13px;">
                                    {improvement}
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    if not agent_match:
                        st.error("📋 No multi-agent analysis data available for this call")
                    elif not agent_match.get('success', False):
                        st.warning("⚠️ Multi-agent system could not complete analysis")
                        if 'error' in agent_match:
                            st.info(f"Error: {agent_match['error']}")
                    elif 'evaluation' not in agent_match:
                        st.info(f"📊 Agent identified as {agent_match.get('agent_name', 'Unknown')}, but evaluation data is incomplete")
                    else:
                        agent_name = agent_match.get('agent_name', 'Unknown')
                        if agent_name.lower() == "unidentified" or agent_name.lower() == "unknown":
                            st.info("⚠️ Multi-agent system could not identify the agent for this call")
                        else:
                            st.info("Multi-agent analysis incomplete for this agent.")
        else:
            st.info("No multi-agent analysis available.")
    
    with tabs[2]:
        st.header("😊 Sentiment Analysis")
        if st.session_state.formatted_conversations and call_options:
        
            conversation = st.session_state.formatted_conversations[st.session_state.selected_call_index]
            
            # Process data for visualization
            if conversation:
                # Create two columns for the chart and analysis
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.subheader("📈 Sentiment Trends")
                    
                    # Prepare data for quarterly view
                    total_entries = len(conversation)
                    quarter_size = max(1, total_entries / 4)
                    
                    # Initialize data structures for quarterly view
                    quarters = ["Q1", "Q2", "Q3", "Q4"]
                    agent_by_quarter = {q: [] for q in quarters}
                    customer_by_quarter = {q: [] for q in quarters}
                    
                    # Initialize counters for sentiment statistics
                    agent_positive, agent_neutral, agent_negative = 0, 0, 0
                    customer_positive, customer_neutral, customer_negative = 0, 0, 0
                    
                    # Process each entry in the conversation
                    for i, entry in enumerate(conversation):
                        # Determine which quarter this entry belongs to
                        quarter_idx = min(3, int(i / quarter_size))
                        quarter = quarters[quarter_idx]
                        
                        # Get sentiment value (ensure uppercase comparison)
                        sentiment = entry.get("sentiment", "NEUTRAL").upper()
                        sentiment_value = 0
                        if sentiment == "POSITIVE":
                            sentiment_value = 1
                        elif sentiment == "NEGATIVE":
                            sentiment_value = -1
                        
                        # Determine if agent or customer and update counters
                        if "agent" in entry.get("speaker", "").lower():
                            agent_by_quarter[quarter].append(sentiment_value)
                            if sentiment == "POSITIVE":
                                agent_positive += 1
                            elif sentiment == "NEGATIVE":
                                agent_negative += 1
                            else:
                                agent_neutral += 1
                        else:
                            customer_by_quarter[quarter].append(sentiment_value)
                            if sentiment == "POSITIVE":
                                customer_positive += 1
                            elif sentiment == "NEGATIVE":
                                customer_negative += 1
                            else:
                                customer_neutral += 1
                    
                    # Calculate average sentiment for each quarter
                    agent_quarterly_sentiment = []
                    customer_quarterly_sentiment = []
                    
                    for quarter in quarters:
                        # Agent sentiment for this quarter
                        if agent_by_quarter[quarter]:
                            agent_avg = sum(agent_by_quarter[quarter]) / len(agent_by_quarter[quarter])
                        else:
                            agent_avg = 0
                        agent_quarterly_sentiment.append(agent_avg)
                        
                        # Customer sentiment for this quarter
                        if customer_by_quarter[quarter]:
                            customer_avg = sum(customer_by_quarter[quarter]) / len(customer_by_quarter[quarter])
                        else:
                            customer_avg = 0
                        customer_quarterly_sentiment.append(customer_avg)
                    
                    # Calculate overall averages
                    agent_avg = sum(agent_quarterly_sentiment) / 4
                    customer_avg = sum(customer_quarterly_sentiment) / 4
                    
                    # Create a DataFrame for plotting
                    chart_data = pd.DataFrame({
                        'Quarter': quarters,
                        'Customer': customer_quarterly_sentiment,
                        'Agent': agent_quarterly_sentiment
                    })
                    
                    # Create chart using Plotly
                    fig = go.Figure()
                    
                    # Add customer sentiment line (purple)
                    fig.add_trace(go.Scatter(
                        x=chart_data['Quarter'],
                        y=chart_data['Customer'],
                        mode='lines+markers',
                        name='Customer',
                        line=dict(color='#9400D3', width=3),
                        marker=dict(size=10)
                    ))
                    
                    # Add agent sentiment line (orange)
                    fig.add_trace(go.Scatter(
                        x=chart_data['Quarter'],
                        y=chart_data['Agent'],
                        mode='lines+markers',
                        name='Agent',
                        line=dict(color='#FFA500', width=3),
                        marker=dict(size=10)
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title="",
                        title_font=dict(size=18),
                        xaxis_title="Call Quarter",
                        yaxis_title="Sentiment Score",
                        yaxis=dict(
                            range=[-1.2, 1.2],
                            tickvals=[-1, 0, 1],
                            ticktext=["Negative", "Neutral", "Positive"],
                            gridcolor='lightgrey'
                        ),
                        xaxis=dict(
                            gridcolor='lightgrey'
                        ),
                        plot_bgcolor='white',
                        hovermode="x unified",
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="right",
                            x=0.99
                        ),
                        margin=dict(t=10, b=10, l=10, r=10)
                    )
                    
                    # Display the plot
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("📊 Sentiment Analysis")
                    
                    # Multi-Agent AI Insights if available
                    if st.session_state.agent_category_matches:
                        agent_match = st.session_state.agent_category_matches[st.session_state.selected_call_index]
                        if agent_match and agent_match.get('success', False):
                            customer_insights = agent_match.get('customer_insights', {})
                            if customer_insights:
                                satisfaction = customer_insights.get('customer_satisfaction_prediction', 'Unknown')
                                satisfaction_color = "#28a745" if satisfaction == "High" else "#ffc107" if satisfaction == "Medium" else "#dc3545"
                                
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px; border-radius: 8px; margin-bottom: 15px;">
                                    <div style="font-size: 14px; margin-bottom: 5px;">🤖 AI Prediction</div>
                                    <div style="font-weight: bold;">Customer Satisfaction: {satisfaction}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Agent Sentiment Section
                    st.markdown("#### 👨‍💼 Agent Sentiment")
                    st.markdown(f"Average Sentiment: **{agent_avg:.2f}**")
                    
                    # Create bullet points with proper spacing
                    st.markdown(f"• Positive statements: **{agent_positive}**")
                    st.markdown(f"• Neutral statements: **{agent_neutral}**")
                    st.markdown(f"• Negative statements: **{agent_negative}**")
                    
                    # Customer Sentiment Section
                    st.markdown("#### 👤 Customer Sentiment")
                    st.markdown(f"Average Sentiment: **{customer_avg:.2f}**")
                    
                    # Create bullet points with proper spacing
                    st.markdown(f"• Positive statements: **{customer_positive}**")
                    st.markdown(f"• Neutral statements: **{customer_neutral}**")
                    st.markdown(f"• Negative statements: **{customer_negative}**")
                    
                    # Call Outcome Analysis
                    st.markdown("#### 📈 Call Outcome")
                    
                    # Determine if sentiment improved from first to last quarter
                    customer_trend = "Stable"
                    if customer_quarterly_sentiment[0] < customer_quarterly_sentiment[3]:
                        customer_trend = "Improving"
                    elif customer_quarterly_sentiment[0] > customer_quarterly_sentiment[3]:
                        customer_trend = "Declining"
                    
                    st.markdown(f"Customer sentiment trend: **{customer_trend}**")
                    
                    # Final sentiment description
                    final_sentiment_text = "Neutral"
                    if customer_quarterly_sentiment[3] > 0.3:
                        final_sentiment_text = "Positive"
                    elif customer_quarterly_sentiment[3] < -0.3:
                        final_sentiment_text = "Negative"
                    
                    st.markdown(f"Final customer sentiment: **{final_sentiment_text}**")
                    
                    # Check for sentiment crossover points
                    crossovers = []
                    for i in range(1, 4):
                        prev_diff = customer_quarterly_sentiment[i-1] - agent_quarterly_sentiment[i-1]
                        curr_diff = customer_quarterly_sentiment[i] - agent_quarterly_sentiment[i]
                        if (prev_diff < 0 and curr_diff > 0) or (prev_diff > 0 and curr_diff < 0):
                            crossovers.append(f"Q{i}")
                    
                    if crossovers:
                        st.markdown("#### 🔄 Key Moments")
                        st.markdown(f"Sentiment crossover at: **{', '.join(crossovers)}**")
                
                # Display conversation flow below the charts
                st.markdown("---")
                st.markdown("#### 💬 Conversation Details")
                
                # Display conversation with sentiment highlighting
                def highlight_sentiment(val):
                    if isinstance(val, str):
                        val = val.upper()
                        if val == "POSITIVE":
                            return "background-color: #d4f1d4"
                        elif val == "NEGATIVE":
                            return "background-color: #f1d4d4"
                        else:
                            return "background-color: #e9ecef"
                    return ""
                
                # Create DataFrame for display
                full_convo = pd.DataFrame({
                    "Utterance": range(1, len(conversation) + 1),
                    "Speaker": [entry.get("speaker", "Unknown") for entry in conversation],
                    "Text": [entry.get("text", "") for entry in conversation],
                    "Sentiment": [entry.get("sentiment", "NEUTRAL") for entry in conversation]
                })
                
                st.dataframe(
                    full_convo.style.applymap(highlight_sentiment, subset=["Sentiment"]),
                    use_container_width=True
                )
        else:
            st.info("No conversation data available for this call.")