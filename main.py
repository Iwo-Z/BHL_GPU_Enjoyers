import streamlit as st
import time
import random
from classifier import *
from textrank import *

class Ollama(object):
    @staticmethod
    def generate(model, prompt):
        if "quantum computing" in prompt.lower():
            response_text = (
                "Based on the summarized prompt, quantum computing utilizes quantum phenomena "
                "like superposition and entanglement for computation. This targeted approach "
                "saves energy by only passing the essential context to the model."
            )
        else:
            response_text = (
                "The LLM received the shortened prompt and generated a relevant response. "
                "The overall process was executed efficiently thanks to the pre-processing layer."
            )
        return {'response': response_text}
ollama = Ollama()

class Evaluation(object):
    """Encapsulates the full optimized LLM pipeline."""

    def __init__(self):
        # Mock initialization for the frontend display
        self.model = DistilBertForSequenceClassification.from_pretrained("model")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("tokenizer")

    def run_LLM(self, prompt):
        # Use the mocked Ollama class
        response = ollama.generate(
            model='mistral',
            prompt=prompt
        )
        return response['response']
    
    def run_textrank(self, prompt):
        # Use the mocked TextRankSummarizer class
        tr = TextRankSummarizer()
        summary = tr.summarize(prompt)
        return summary
    
    def run_optimized_LLM(self, raw_prompt):
        """
        Executes the optimized pipeline and collects metrics for the frontend.
        """
        cl = Classifier(self.model, self.tokenizer)
        class_ = cl.predict(raw_prompt)
        
        # Initialize metrics
        original_length = len(raw_prompt)
        processed_length = original_length
        reduction = 0.0
        is_greeting = False
        processed_prompt = raw_prompt
        final_response = "Error: Unknown classification result." # Default

        if class_ != "prompt": 
            # Case 1: Simple prompt (Greeting/Goodbye) -> Skip LLM
            is_greeting = True
            final_response = "Hello there! How can I help you today?"
            
        elif class_ == "prompt": 
            # Case 2: Complex prompt -> Summarize and call LLM
            
            # 1. Summarization/Shortening
            summarized_prompt = self.run_textrank(raw_prompt)
            processed_prompt = summarized_prompt
            
            # Calculate metrics
            processed_length = len(processed_prompt)
            if original_length > 0 and processed_length < original_length:
                reduction = ((original_length - processed_length) / original_length) * 100
            
            # 2. LLM Call
            final_response = self.run_LLM(processed_prompt)
            
        else:
            # Fallback
            final_response = "Error: Classification logic returned an unexpected result."


        return {
            "final_response": final_response,
            "is_greeting": is_greeting,
            "processed_prompt": processed_prompt,
            "original_length": original_length,
            "processed_length": processed_length,
            "reduction_percent": round(reduction, 1)
        }

# =================================================================
# --- STREAMLIT FRONTEND LOGIC ---
# =================================================================

def run_processing_pipeline(raw_prompt):
    """
    Instantiates the Evaluation object and runs the optimized pipeline.
    Updates the results in Streamlit's session state.
    """
    
    # Simulate setup delay
    time.sleep(0.5) 
    
    # Execute the user's core logic
    evaluator = Evaluation()
    processing_results = evaluator.run_optimized_LLM(raw_prompt)
    
    # Unpack results into session state
    st.session_state.is_greeting = processing_results["is_greeting"]
    st.session_state.processed_prompt = processing_results["processed_prompt"]
    st.session_state.original_length = processing_results["original_length"]
    st.session_state.processed_length = processing_results["processed_length"]
    st.session_state.reduction_percent = processing_results["reduction_percent"]
    st.session_state.ai_response = processing_results["final_response"]


def main():
    st.set_page_config(
        page_title="AI Prompt Processor Prototype",
        layout="centered",
    )

    st.title("🧠 Efficient AI Pipeline Demo")
    st.markdown("This application demonstrates a tiered architecture for **cost and energy efficiency** using custom classification and summarization.")

    # --- SESSION STATE INITIALIZATION ---
    if "is_greeting" not in st.session_state:
        st.session_state.is_greeting = False
    if "processed_prompt" not in st.session_state:
        st.session_state.processed_prompt = ""
    if "ai_response" not in st.session_state:
        st.session_state.ai_response = ""
    if "original_length" not in st.session_state:
        st.session_state.original_length = 0
    if "processed_length" not in st.session_state:
        st.session_state.processed_length = 0
    if "reduction_percent" not in st.session_state:
        st.session_state.reduction_percent = 0.0
    # --- END SESSION STATE INITIALIZATION ---
    
    # --- SIDEBAR: Flow Visualization ---
    with st.sidebar:
        st.subheader("AI Pipeline Flow")
        st.markdown("""
        The pipeline reduces costs and energy by:
        1. **Classifying** simple prompts using a small, local model (DistilBERT).
        2. **Summarizing** complex prompts using TextRank before calling the LLM.
        
        | Step | Logic Implemented | Action Result |
        | :--- | :--- | :--- |
        | **1. Input** | User enters prompt | Raw Text |
        | **2. Classify** | `Classifier.predict()` | `hardcode` (Skip LLM) / `prompt` (Continue) |
        | **3. Summarize** | `TextRankSummarizer.summarize()` | Prompt Shortening |
        | **4. LLM Call** | `ollama.generate('mistral', ...)` | Final Response |
        """)
    # --- END SIDEBAR ---

    st.subheader("1. Enter Your Raw Prompt")
    
    # Input Area
    raw_prompt = st.text_area(
        "Input Prompt", 
        placeholder="e.g., Hi, I need a very detailed, multi-paragraph explanation of quantum computing, including its history and latest developments.", 
        height=150,
        key="current_raw_prompt_text_area"
    )
    
    # Action Button
    if st.button("Run Optimized Pipeline", use_container_width=True, type="primary"):
        if not raw_prompt:
            st.error("Please enter a prompt to start the pipeline.")
        else:
            with st.spinner("Executing optimized pipeline..."):
                run_processing_pipeline(raw_prompt)
                st.success("Pipeline executed successfully!")
    
    st.divider()

    # --- RESULTS DISPLAY AND VISUALIZATION ---
    
    if st.session_state.ai_response:
        
        st.subheader("2. Pipeline Status and Efficiency")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.is_greeting:
                st.metric("Prompt Classified As", "HARDCODED (Greeting)", "LLM SKIPPED")
            else:
                st.metric("Prompt Classified As", "COMPLEX (Needs LLM)", "LLM CALLED")
                
        with col2:
            st.metric("Reduction (%)", f"{st.session_state.reduction_percent}%", "Text Summarized")

        with col3:
            if st.session_state.is_greeting:
                st.metric("Efficiency Gain", "Tier 0 (Zero Cost)", "Max Savings")
            else:
                st.metric("Efficiency Gain", "Tier 1 (Optimized Cost)", f"Saved {st.session_state.reduction_percent}% Tokens")

        st.markdown("---")

        if not st.session_state.is_greeting:
            st.subheader("3. Processed Prompt (LLM Input)")
            
            # Display lengths for complex prompts
            st.text(f"Original Length: {st.session_state.original_length} characters")
            st.text(f"Processed Length: {st.session_state.processed_length} characters")
            
            st.code(st.session_state.processed_prompt, language='text')

            st.info("The original prompt was summarized by TextRank before being sent to the LLM (Ollama/Mistral), reducing tokens and consumption.")
        
        st.subheader("4. Final AI Response")
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #10B981;">
                {st.session_state.ai_response}
            </div>
            """, 
            unsafe_allow_html=True
        )


if __name__ == '__main__':
    main()