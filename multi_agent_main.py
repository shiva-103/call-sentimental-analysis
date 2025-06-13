import os
import sys
import json
import time
import requests
import traceback
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from assemblyai import TranscriptionConfig, Transcriber
from typing import Dict, Any, Optional, List
import assemblyai as aai
import streamlit as st
import autogen
from autogen import ConversableAgent, UserProxyAgent

# API keys and configuration
aai.settings.api_key = st.secrets['AAI_API']
EMAIL_ADDRESS = st.secrets['EMAIL_USER']
EMAIL_PASSWORD = st.secrets['EMAIL_PASSWORD']
GROQ_API_KEY = st.secrets['GROQ_API_KEY']
GROQ_MODEL = "llama3-8b-8192"

# Configure Groq LLM for Autogen
config_list = [
    {
        "model": GROQ_MODEL,
        "api_key": GROQ_API_KEY,
        "base_url": "https://api.groq.com/openai/v1",
        "api_type": "openai",
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.1,
    "timeout": 120,
}

# Configure transcription settings
config = aai.TranscriptionConfig(
    speaker_labels=True,
    sentiment_analysis=True,
    entity_detection=True,
)

AGENT_PROFILES = {
    "Andrew K": {
        "id": "AK001",
        "name": "Andrew K",
        "categories": ["technical_support", "product_information"],
        "expertise_level": "senior",
        "department": "technical_support"
    },
    "Sarah M": {
        "id": "SM002",
        "name": "Sarah M",
        "categories": ["billing", "account_management"],
        "expertise_level": "mid",
        "department": "customer_service"
    },
    "Michael J": {
        "id": "MJ003",
        "name": "Michael J",
        "categories": ["sales", "product_upsell"],
        "expertise_level": "senior",
        "department": "sales"
    },
    "Lisa P": {
        "id": "LP004",
        "name": "Lisa P",
        "categories": ["returns", "order_management"],
        "expertise_level": "junior",
        "department": "customer_service"
    }
}

CONVERSATION_CATEGORIES = [
    "technical_support",
    "billing",
    "sales",
    "account_management",
    "product_information",
    "returns",
    "order_management"
]

# Simple decorator for tracking function execution time
def track_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"⏱️ Starting: {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"⏱️ Completed: {func.__name__} - Took {execution_time:.4f} seconds")
        return result
    return wrapper

class MultiAgentCallAnalysisSystem:
    def __init__(self):
        self.transcripts = []
        self.call_summary = {}
        self.agent_analysis = {}
        self.agent_evaluation = None
        self.sentiment_results = []
        self.formatted_conversations = []
        
        # Initialize Autogen agents
        self._setup_agents()
    
    def _setup_agents(self):
        """Setup the multi-agent system with specialized agents"""
        
        # 1. Transcript Analyst Agent - Handles transcript analysis and categorization
        self.transcript_analyst = ConversableAgent(
            name="TranscriptAnalyst",
            system_message="""You are a Transcript Analyst specializing in customer service call analysis. 
            Your responsibilities:
            1. Analyze transcripts to identify agents and conversation categories
            2. Generate comprehensive call summaries
            3. Extract key insights from customer-agent interactions
            
            Always provide responses in valid JSON format when requested.
            Categories available: technical_support, billing, sales, account_management, product_information, returns, order_management
            Agent names available: Andrew K, Sarah M, Michael J, Lisa P
            
            Be precise and thorough in your analysis.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        
        # 2. Agent Performance Evaluator - Evaluates agent performance against standards
        self.performance_evaluator = ConversableAgent(
            name="PerformanceEvaluator",
            system_message="""You are an Agent Performance Evaluator specializing in customer service quality assessment.
            Your responsibilities:
            1. Evaluate agent performance against company standards
            2. Assess agent-category match appropriateness
            3. Rate agent empathy, professionalism, and effectiveness
            4. Identify strengths and areas for improvement
            
            Company Standards to evaluate:
            - Used customer's name minimum once
            - Active listening (remembers info)
            - Does not interrupt
            - Used apology & empathy where required
            - Used Please/Thank you appropriately
            - Transferred to correct department if needed
            - Provided alternatives
            - Maintained proper tone
            - Verified customer appropriately
            - Provided correct information
            - Tagged call properly in CRM
            
            Rate each standard as Yes/No/N/A and provide overall rating 1-10.
            Always respond in valid JSON format when requested.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        
        # 3. Customer Insights Analyst - Analyzes customer needs and predicts next actions
        self.customer_insights = ConversableAgent(
            name="CustomerInsights",
            system_message="""You are a Customer Insights Analyst specializing in understanding customer needs and behavior.
            Your responsibilities:
            1. Identify underlying customer needs that may not have been directly addressed
            2. Predict next best actions for customer success
            3. Analyze customer pain points and satisfaction levels
            4. Recommend proactive follow-up strategies
            
            Focus on actionable insights that can improve customer experience.
            Keep recommendations concise (10-20 words each).
            Always respond in valid JSON format when requested.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        
        # 4. Quality Assurance Manager - Determines intervention needs and coordinates responses
        self.qa_manager = ConversableAgent(
            name="QAManager",
            system_message="""You are a Quality Assurance Manager responsible for escalation decisions and quality oversight.
            Your responsibilities:
            1. Determine if human intervention is needed based on call analysis
            2. Decide escalation level (urgent_email, high_priority_ticket, normal_ticket, none)
            3. Provide specific, actionable recommendations for improvement
            4. Make final decisions on customer follow-up requirements
            
            Escalation criteria:
            - urgent_email: Unresolved + negative sentiment + callback needed, or critical issues (security, billing errors, outages)
            - high_priority_ticket: Unresolved + negative sentiment, or problematic agent interactions
            - normal_ticket: Unresolved but customer not negative, or callback needed for resolved issues
            - none: Resolved, positive sentiment, no callbacks needed
            
            When providing recommended_actions, ensure each action is:
            - Specific and actionable (10-15 words)
            - Addresses identified issues from the analysis
            - Can be implemented by customer service teams
            
            ALWAYS respond with valid JSON format. NEVER include explanatory text outside the JSON structure.
            The recommended_actions field must be a proper JSON array of strings.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        
        # 5. System Coordinator - Orchestrates the multi-agent workflow
        self.coordinator = UserProxyAgent(
            name="SystemCoordinator",
            system_message="""You are the System Coordinator responsible for orchestrating the multi-agent analysis workflow.
            You will coordinate between all agents to ensure comprehensive call analysis.""",
            human_input_mode="NEVER",
            code_execution_config=False,
        )

    def parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Helper function to parse JSON from agent responses with improved error handling"""
        try:
            # Try direct JSON parsing first
            parsed = json.loads(response_text.strip())
            
            # Validate that recommended_actions is a proper list
            if 'recommended_actions' in parsed:
                if isinstance(parsed['recommended_actions'], str):
                    # Try to convert string to list
                    actions_str = parsed['recommended_actions']
                    if actions_str.startswith('[') and actions_str.endswith(']'):
                        try:
                            parsed['recommended_actions'] = json.loads(actions_str)
                        except:
                            # Split by common delimiters
                            import re
                            actions = re.split(r'[,;\n]\s*', actions_str.strip('[]'))
                            parsed['recommended_actions'] = [action.strip().strip('"\'') for action in actions if action.strip()]
                    else:
                        # Split by common delimiters
                        import re
                        actions = re.split(r'[,;\n]\s*', actions_str)
                        parsed['recommended_actions'] = [action.strip().strip('"\'') for action in actions if action.strip()]
            
            return parsed
            
        except json.JSONDecodeError:
            # Extract from code blocks if needed
            if "```json" in response_text and "```" in response_text:
                json_content = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_content = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_content = response_text
            
            try:
                parsed = json.loads(json_content)
                
                # Validate recommended_actions again
                if 'recommended_actions' in parsed and isinstance(parsed['recommended_actions'], str):
                    actions_str = parsed['recommended_actions']
                    import re
                    actions = re.split(r'[,;\n]\s*', actions_str.strip('[]'))
                    parsed['recommended_actions'] = [action.strip().strip('"\'') for action in actions if action.strip()]
                
                return parsed
                
            except json.JSONDecodeError:
                # Fallback: try to extract key-value pairs
                extracted_data = {}
                lines = response_text.replace("```", "").strip().split("\n")
                
                for line in lines:
                    if ":" in line:
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            key = parts[0].strip().strip('"{}').strip()
                            value = parts[1].strip().strip('",').strip()
                            
                            if key and value:
                                # Special handling for recommended_actions
                                if key == "recommended_actions" and '[' in value:
                                    try:
                                        # Try to parse as JSON array
                                        extracted_data[key] = json.loads(value)
                                    except:
                                        # Parse manually
                                        import re
                                        actions = re.findall(r'"([^"]*)"', value)
                                        if not actions:
                                            actions = re.split(r'[,;\n]\s*', value.strip('[]'))
                                        extracted_data[key] = [action.strip() for action in actions if action.strip()]
                                else:
                                    # Check if the value looks like a list
                                    if value.startswith('[') and value.endswith(']'):
                                        try:
                                            list_value = json.loads(value)
                                            if isinstance(list_value, list):
                                                extracted_data[key] = list_value
                                                continue
                                        except:
                                            pass
                                    extracted_data[key] = value
                
                if extracted_data:
                    return extracted_data
                
                # If all parsing attempts fail, raise the original error
                raise json.JSONDecodeError(
                    f"Failed to parse agent response as JSON: {response_text[:100]}...", 
                    response_text, 0
                )

    @track_time
    def upload_audio_files(self, file_paths):
        """Upload multiple audio files for processing"""
        print("Uploading and transcribing audio files...")
        transcripts = []
        
        for file_path in file_paths:
            try:
                file_start = time.time()
                transcript = aai.Transcriber().transcribe(
                    file_path,
                    config=config
                )
                file_end = time.time()
                transcripts.append(transcript)
                print(f"Transcribed: {file_path} in {file_end - file_start:.4f}s")
            except Exception as e:
                print(f"Error transcribing {file_path}: {str(e)}")
                
        self.transcripts = transcripts
        return transcripts

    @track_time
    def get_text_transcripts(self):
        """Extract and format text from transcripts with speaker information"""
        formatted_transcripts = []
        
        for i, transcript in enumerate(self.transcripts):
            start_time = time.time()
            if not transcript.utterances:
                formatted_text = transcript.text
            else:
                formatted_text = "\n".join(
                    f"Speaker {utterance.speaker}: {utterance.text}" 
                    for utterance in transcript.utterances
                )
            
            formatted_transcripts.append(formatted_text)
            end_time = time.time()
            print(f"Processed transcript {i+1} in {end_time - start_time:.4f}s")
            
        return formatted_transcripts

    @track_time
    def perform_sentiment_analysis(self):
        """Extract sentiment analysis from AssemblyAI transcripts"""
        sentiment_results = []
        
        for i, transcript in enumerate(self.transcripts):
            if transcript.sentiment_analysis:
                sentiment_results.append(transcript.sentiment_analysis)
                print(f"Extracted sentiment from transcript {i+1}")
                
        self.sentiment_results = sentiment_results
        return sentiment_results

    @track_time
    def format_sentiment_data(self, sentiment_results):
        """Format sentiment analysis results for monitoring purposes"""
        format_start = time.time()
        formatted_conversations = []
        
        for i, result_set in enumerate(sentiment_results):
            conversation = [
                {
                    "speaker": f"Speaker {sentiment.speaker}",
                    "text": sentiment.text,
                    "sentiment": str(sentiment.sentiment).replace("SentimentType.", "").replace("'", "").replace(">", "")
                }
                for sentiment in result_set
            ]
            formatted_conversations.append(conversation)
            
        format_end = time.time()
        print(f"⏱️ Sentiment formatting: {format_end - format_start:.4f}s")
        self.formatted_conversations = formatted_conversations
        return formatted_conversations

    @track_time
    def multi_agent_analysis(self, transcript_idx=0):
        """Comprehensive multi-agent analysis workflow"""
        if not self.transcripts or transcript_idx >= len(self.transcripts):
            print("No transcript available for analysis")
            return None
            
        # Get the formatted transcript
        transcript_text = self.get_text_transcripts()[transcript_idx]
        
        print(f"\n🤖 Starting Multi-Agent Analysis for Call {transcript_idx + 1}")
        
        try:
            # Step 1: Agent Identification and Category Analysis
            print("\n⏱️ STEP 1: AGENT IDENTIFICATION & CATEGORIZATION")
            step_start = time.time()
            
            identification_prompt = f"""
            Analyze this customer service call transcript and identify:
            1. The agent's name (from: {list(AGENT_PROFILES.keys())})
            2. The conversation category (from: {CONVERSATION_CATEGORIES})
            
            TRANSCRIPT:
            {transcript_text}
            
            Return ONLY a JSON object with:
            {{
                "agent_name": "agent name or 'Unidentified'",
                "category": "conversation category"
            }}
            """
            
            identification_response = self.coordinator.initiate_chat(
                self.transcript_analyst,
                message=identification_prompt,
                max_turns=1,
                silent=True
            )
            
            # Extract the last message from transcript_analyst
            identification_result = identification_response.chat_history[-1]['content']
            identification_data = self.parse_json_response(identification_result)
            
            agent_name = identification_data.get("agent_name", "Unidentified")
            category = identification_data.get("category", "unknown")
            
            step_end = time.time()
            print(f"⏱️ Step 1 completed in {step_end - step_start:.4f}s")
            print(f"Identified Agent: {agent_name}, Category: {category}")
            
            # Step 2: Call Summary Generation
            print("\n⏱️ STEP 2: CALL SUMMARY GENERATION")
            step_start = time.time()
            
            summary_prompt = f"""
            Generate a comprehensive call summary for this transcript:
            
            TRANSCRIPT:
            {transcript_text}
            
            Return a JSON object with:
            {{
                "Summary": "brief 1-2 sentence overview",
                "Topic": "main subject",
                "Product": "product/service discussed",
                "Resolved": "Yes/No/Partial",
                "Callback": "Yes/No",
                "Politeness": "Low/Medium/High",
                "Customer sentiment": "Negative/Neutral/Positive",
                "Agent sentiment": "Negative/Neutral/Positive",
                "Action": "actions taken by agent"
            }}
            """
            
            summary_response = self.coordinator.initiate_chat(
                self.transcript_analyst,
                message=summary_prompt,
                max_turns=1,
                silent=True
            )
            
            summary_result = summary_response.chat_history[-1]['content']
            call_summary = self.parse_json_response(summary_result)
            self.call_summary = call_summary
            
            step_end = time.time()
            print(f"⏱️ Step 2 completed in {step_end - step_start:.4f}s")
            
            # Step 3: Agent Performance Evaluation
            print("\n⏱️ STEP 3: AGENT PERFORMANCE EVALUATION")
            step_start = time.time()
            
            # Check if agent is authorized for category
            is_authorized = False
            agent_profile = {}
            if agent_name in AGENT_PROFILES:
                agent_profile = AGENT_PROFILES[agent_name]
                is_authorized = category in agent_profile.get("categories", [])
            
            evaluation_prompt = f"""
            Evaluate agent performance for this call:
            
            AGENT: {agent_name}
            CATEGORY: {category}
            AUTHORIZED: {is_authorized}
            AGENT PROFILE: {json.dumps(agent_profile, indent=2)}
            
            TRANSCRIPT:
            {transcript_text}
            
            Evaluate against company standards and return JSON:
            {{
                "standards_met": {{
                    "Used the customer's name minimum once on the call": "Yes/No/N/A",
                    "Does Active Listening (remembers info)": "Yes/No/N/A",
                    "Does not interrupt": "Yes/No/N/A",
                    "Used apology & empathy wherever required": "Yes/No/N/A",
                    "Used Please / Thank you wherever appropriate": "Yes/No/N/A",
                    "Transferred to correct department": "Yes/No/N/A",
                    "Did the CSA provide alternatives": "Yes/No/N/A",
                    "Did the CSA maintain proper tone throughout the call": "Yes/No/N/A",
                    "Verified the customer appropriately as per the nature of the call": "Yes/No/N/A",
                    "Provided correct information": "Yes/No/N/A",
                    "Tagged the call properly in the CRM": "Yes/No/N/A"
                }},
                "strengths": ["strength 1", "strength 2", "strength 3"],
                "areas_for_improvement": ["improvement 1", "improvement 2", "improvement 3"],
                "overall_rating": 8,
                "category_expertise": "High/Medium/Low",
                "customer_pain_points": ["pain point 1", "pain point 2"],
                "resolution_quality": "Yes/No/Partial - satisfaction rating 1-10",
                "agent_empathy": 8,
                "would_recommend": "Yes/No/Maybe",
                "next_best_actions": ["action 1", "action 2"]
            }}
            """
            
            evaluation_response = self.coordinator.initiate_chat(
                self.performance_evaluator,
                message=evaluation_prompt,
                max_turns=1,
                silent=True
            )
            
            evaluation_result = evaluation_response.chat_history[-1]['content']
            agent_evaluation = self.parse_json_response(evaluation_result)
            
            step_end = time.time()
            print(f"⏱️ Step 3 completed in {step_end - step_start:.4f}s")
            
            # Step 4: Customer Insights Analysis
            print("\n⏱️ STEP 4: CUSTOMER INSIGHTS ANALYSIS")
            step_start = time.time()
            
            insights_prompt = f"""
            Analyze customer needs and predict next actions:
            
            CALL SUMMARY: {json.dumps(call_summary, indent=2)}
            
            TRANSCRIPT:
            {transcript_text}
            
            Return JSON with:
            {{
                "underlying_needs": ["need 1", "need 2", "need 3"],
                "next_best_actions": ["action 1", "action 2", "action 3"],
                "customer_satisfaction_prediction": "High/Medium/Low",
                "follow_up_priority": "High/Medium/Low"
            }}
            """
            
            insights_response = self.coordinator.initiate_chat(
                self.customer_insights,
                message=insights_prompt,
                max_turns=1,
                silent=True
            )
            
            insights_result = insights_response.chat_history[-1]['content']
            customer_insights = self.parse_json_response(insights_result)
            
            step_end = time.time()
            print(f"⏱️ Step 4 completed in {step_end - step_start:.4f}s")
            
            # Step 5: Quality Assurance and Intervention Decision
            print("\n⏱️ STEP 5: QA INTERVENTION DECISION")
            step_start = time.time()
            
            qa_prompt = f"""
            As a Quality Assurance Manager, determine intervention level and provide specific actionable recommendations based on this comprehensive analysis:
            
            CALL SUMMARY: {json.dumps(call_summary, indent=2)}
            AGENT EVALUATION: {json.dumps(agent_evaluation, indent=2)}
            CUSTOMER INSIGHTS: {json.dumps(customer_insights, indent=2)}
            
            Based on the analysis above, provide detailed recommendations and intervention decision.
            
            INTERVENTION CRITERIA:
            - urgent_email: Critical issues, unresolved + negative sentiment + callback needed, or safety/security concerns
            - high_priority_ticket: Unresolved + negative sentiment, poor agent performance, authorization mismatches
            - normal_ticket: Standard follow-up needed, partially resolved issues, neutral sentiment
            - none: Fully resolved, positive outcome, no further action needed
            
            Return ONLY a valid JSON object with this exact structure:
            {{
                "intervention_type": "urgent_email or high_priority_ticket or normal_ticket or none",
                "reasoning": "Clear explanation for the intervention decision based on specific findings",
                "priority_level": "High or Medium or Low",
                "recommended_actions": [
                    "Specific actionable recommendation 1 (10-15 words)",
                    "Specific actionable recommendation 2 (10-15 words)",
                    "Specific actionable recommendation 3 (10-15 words)"
                ]
            }}
            
            Ensure recommended_actions are specific, actionable steps that address the identified issues.
            """
            
            qa_response = self.coordinator.initiate_chat(
                self.qa_manager,
                message=qa_prompt,
                max_turns=1,
                silent=True
            )
            
            qa_result = qa_response.chat_history[-1]['content']
            qa_decision = self.parse_json_response(qa_result)
            
            step_end = time.time()
            print(f"⏱️ Step 5 completed in {step_end - step_start:.4f}s")
            
            # Compile final analysis result
            final_result = {
                "success": True,
                "agent_name": agent_name,
                "agent_profile": agent_profile,
                "category": category,
                "is_authorized": is_authorized,
                "authorization_details": f"Agent {agent_name} {'is' if is_authorized else 'is not'} authorized for {category}",
                "evaluation": agent_evaluation,
                "call_summary": call_summary,
                "customer_insights": customer_insights,
                "qa_decision": qa_decision,
                "timestamp": time.time()
            }
            
            # Store results
            self.agent_analysis = final_result
            
            # Send email if intervention needed
            intervention_type = qa_decision.get("intervention_type", "none")
            if intervention_type != "none":
                self.send_email_alert("lukkashivacharan@gmail.com", intervention_type)
            
            print(f"\n✅ Multi-Agent Analysis Complete")
            return final_result
            
        except Exception as e:
            print(f"Error in multi-agent analysis: {str(e)}")
            print(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "stage": "multi_agent_analysis"
            }

    def check_human_intervention(self):
        """Determine the type of intervention needed based on call summary analysis"""
        if not self.call_summary:
            return "urgent_email"
        
        is_resolved = self.call_summary.get("Resolved", "").lower() in ["yes", "true"]
        is_partial = self.call_summary.get("Resolved", "").lower() == "partial"
        needs_callback = self.call_summary.get("Callback", "").lower() in ["yes", "true"]
        customer_sentiment = self.call_summary.get("Customer sentiment", "").lower()
        agent_sentiment = self.call_summary.get("Agent sentiment", "").lower()
        agent_politeness = self.call_summary.get("Politeness", "").lower()
        topic = self.call_summary.get("Topic", "").lower()
        
        # Urgent intervention
        if any([
            (not is_resolved and customer_sentiment == "negative" and needs_callback),
            (is_partial and customer_sentiment == "negative" and needs_callback),
            (agent_politeness == "low" and customer_sentiment == "negative"),
            any(critical in topic for critical in ["billing error", "outage", "security", "fraud", "legal"])
        ]):
            return "urgent_email"
            
        # High priority ticket
        elif any([
            (not is_resolved and customer_sentiment == "negative"),
            (is_partial and customer_sentiment == "negative"),
            (agent_sentiment == "negative" and customer_sentiment == "negative"),
            (agent_politeness == "medium" and customer_sentiment == "negative"),
            (needs_callback and customer_sentiment != "positive")
        ]):
            return "high_priority_ticket"
            
        # Normal ticket
        elif any([
            (not is_resolved or is_partial) and customer_sentiment != "negative",
            (is_resolved and needs_callback),
            (customer_sentiment == "neutral")
        ]):
            return "normal_ticket"
            
        return "none"

    @track_time
    def send_email_alert(self, recipient_email, intervention_type):
        """Send email alert for human intervention"""
        if not self.call_summary:
            return False
            
        try:
            msg = MIMEMultipart()
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = recipient_email
            
            if intervention_type == "urgent_email":
                priority = "urgent"
                msg['Subject'] = f"URGENT: Human Intervention Needed - Ticket #{id(self.transcripts[0])}"
            elif intervention_type == "high_priority_ticket":
                priority = "high"
                msg['Subject'] = f"HIGH PRIORITY: Ticket #{id(self.transcripts[0])}"
            else:
                priority = "normal"
                msg['Subject'] = f"NORMAL: Ticket #{id(self.transcripts[0])}"
            
            # Prepare agent information if available
            agent_info = ""
            if hasattr(self, 'agent_analysis') and self.agent_analysis and self.agent_analysis.get('success', False):
                agent_data = self.agent_analysis
                agent_evaluation = agent_data.get('evaluation', {})
                
                agent_info = f"""
                <h3>Agent Information:</h3>
                <ul>
                  <li><strong>Agent Name:</strong> {agent_data.get('agent_name', 'Unknown')}</li>
                  <li><strong>Department:</strong> {agent_data.get('agent_profile', {}).get('department', 'Unknown')}</li>
                  <li><strong>Expertise Level:</strong> {agent_data.get('agent_profile', {}).get('expertise_level', 'Unknown')}</li>
                  <li><strong>Call Category:</strong> {agent_data.get('category', 'Unknown')}</li>
                  <li><strong>Agent is Authorized:</strong> <span style="color:{'green' if agent_data.get('is_authorized', False) else 'red'}">{'Yes' if agent_data.get('is_authorized', False) else 'No'}</span></li>
                  <li><strong>Overall Rating:</strong> {agent_evaluation.get('overall_rating', 'N/A')}/10</li>
                </ul>
                
                <h4>Multi-Agent Analysis Results:</h4>
                <ul>
                  <li><strong>QA Decision:</strong> {agent_data.get('qa_decision', {}).get('reasoning', 'N/A')}</li>
                  <li><strong>Customer Satisfaction:</strong> {agent_data.get('customer_insights', {}).get('customer_satisfaction_prediction', 'N/A')}</li>
                </ul>
                """
            
            body = f"""
            <html>
              <body>
                <h2>{"URGENT INTERVENTION" if priority == "urgent" else "HIGH PRIORITY" if priority == "high" else "NORMAL"} TICKET</h2>
                <p>A customer service call has been analyzed by our multi-agent system.</p>
                
                <h3>Ticket Information:</h3>
                <ul>
                  <li><strong>Ticket #:</strong> {id(self.transcripts[0])}</li>
                  <li><strong>Priority:</strong> <span style="color:{'red' if priority == 'urgent' else ('orange' if priority == 'high' else 'green')}"><strong>{priority.upper()}</strong></span></li>
                  <li><strong>Topic:</strong> {self.call_summary.get('Topic', 'Unknown')}</li>
                  <li><strong>Product:</strong> {self.call_summary.get('Product', 'Unknown')}</li>
                  <li><strong>Resolved:</strong> {self.call_summary.get('Resolved', 'No')}</li>
                  <li><strong>Callback:</strong> {self.call_summary.get('Callback', 'Unknown')}</li>
                  <li><strong>Customer sentiment:</strong> {self.call_summary.get('Customer sentiment', 'Unknown')}</li>
                  <li><strong>Agent sentiment:</strong> {self.call_summary.get('Agent sentiment', 'Unknown')}</li>
                </ul>
                
                {agent_info}
                
                <p>Please {"review immediately and take action" if priority == "urgent" else "address as high priority" if priority == "high" else "address at normal priority"}.</p>
              </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Setup SMTP server
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            
            # Send email
            server.send_message(msg)
            server.quit()
            
            print(f"{priority.capitalize()} email alert sent to {recipient_email}")
            return True
            
        except Exception as e:
            print(f"Error sending email alert: {str(e)}")
            return False

    # Legacy method compatibility
    def analyze_agent_category_match(self, transcript_idx=0):
        """Legacy compatibility method - redirects to multi-agent analysis"""
        return self.multi_agent_analysis(transcript_idx)
    
    def analyze_call(self, transcript_text):
        """Legacy compatibility method for call analysis"""
        if hasattr(self, 'call_summary') and self.call_summary:
            return self.call_summary
        return None
    
    def predict_customer_needs(self, transcript_text, call_summary=None):
        """Legacy compatibility method for customer needs prediction"""
        if hasattr(self, 'agent_analysis') and self.agent_analysis:
            customer_insights = self.agent_analysis.get('customer_insights', {})
            return {
                "success": True,
                "predictions": {
                    "underlying_needs": customer_insights.get('underlying_needs', []),
                    "next_best_actions": customer_insights.get('next_best_actions', [])
                }
            }
        return {"success": False, "error": "No analysis available"}
