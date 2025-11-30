import math
import streamlit as st
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from utils.classifier import *
from utils.textrank import *
from codecarbon import EmissionsTracker
import pandas as pd
import requests
import time
import json
import os
from dotenv import load_dotenv

ENERGY_PER_WORD = 0.0000015



load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
MAX_RETRIES = 5

def generate_content_with_retry(prompt: str, system_prompt: str) -> dict:
    if not API_KEY:
        print("Error: API_KEY is missing. Please set your API key.")
        return {}

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    headers = {'Content-Type': 'application/json'}
    
    for attempt in range(MAX_RETRIES):
        try:

            full_url = f"{API_URL}?key={API_KEY}"
            
            response = requests.post(
                full_url, 
                headers=headers, 
                data=json.dumps(payload),
                timeout=30
            )
            response.raise_for_status()
            
            return response.json()

        except requests.exceptions.HTTPError as e:
            if response.status_code in [429, 500, 503]:
                delay = 2 ** attempt + 1
                time.sleep(delay)
            else:
                return {}
        
        except requests.exceptions.RequestException as e:
            delay = 2 ** attempt + 1
            time.sleep(delay)

    return {}


def ask_prompt(user_prompt):
    system_instruction = (
        "You are an extremely fast and concise AI assistant. Your responses must be "
        "direct, short, and immediately answer the user's query without any preamble or fluff. "
    )

    
    api_response = generate_content_with_retry(user_prompt, system_instruction)
    
    if api_response:
        try:
            text_response = api_response['candidates'][0]['content']['parts'][0]['text']
            return text_response
        except (KeyError, IndexError) as e:
            print(f"Error parsing response structure: {e}")
    else:
        print("Could not retrieve content from the API.")

    

class Evaluation(object):
    def __init__(self):
        self.model = DistilBertForSequenceClassification.from_pretrained("model")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("tokenizer")

    def run_LLM(self, prompt):
        # return ask_prompt(prompt)
        tokenizerLLM = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        modelLLM = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            device_map="auto",
            load_in_4bit=True,
            dtype=torch.float16
        )

        inputs = tokenizerLLM(prompt, return_tensors="pt").to(modelLLM.device)
        output = modelLLM.generate(**inputs, max_new_tokens=100)
        return tokenizerLLM.decode(output[0], skip_special_tokens=True)
    
    def run_textrank(self, prompt):
        tr = TextRankSummarizer()
        summary = tr.summarize(prompt)
        return summary
    
    def run_optimized_LLM(self, raw_prompt):
        cl = Classifier(self.model, self.tokenizer)
        tracker = EmissionsTracker()
        tracker.start()

        class_ = cl.predict(raw_prompt)
        
        original_length = len(raw_prompt)
        processed_length = original_length
        reduction = 0.0
        is_greeting = False
        processed_prompt = raw_prompt
        final_response = "Error: Unknown classification result."

        if class_ == "greetings":
            is_greeting = True
            final_response = "Hello there! How can I help you today?"
        elif class_ == "thanking":
            final_response = "No problem! If you have any more questions, feel free to ask."
        elif class_ == "goodbye":
            is_greeting = True
            final_response = "Goodbye! Have a great day!"
            
        elif class_ == "prompt": 
            summarized_prompt = self.run_textrank(raw_prompt)
            processed_prompt = summarized_prompt
            
            processed_length = len(processed_prompt)
            if original_length > 0 and processed_length < original_length:
                reduction = ((original_length - processed_length) / original_length) * 100
            
            final_response = self.run_LLM(processed_prompt)
            
        else:
            final_response = "Error: Classification logic returned an unexpected result."

        emissions = tracker.stop()

        return {
            "final_response": final_response,
            "is_greeting": is_greeting,
            "processed_prompt": processed_prompt,
            "original_length": original_length,
            "processed_length": processed_length,
            "reduction_percent": round(reduction, 1),
            "emissions": round(emissions, 1)
        }

def run_processing_pipeline(raw_prompt):
    time.sleep(0.5) 

    tracker = EmissionsTracker()
    tracker.start()
    
    evaluator = Evaluation()
    processing_results = evaluator.run_optimized_LLM(raw_prompt)
    emissions1 = processing_results["emissions"]

    tracker = EmissionsTracker()
    tracker.start()

    answer_simple = evaluator.run_LLM(raw_prompt)
    emissions2 = tracker.stop()

    df = pd.read_csv("emissions.csv")
    df["energy_consumed"] = df["energy_consumed"].astype(float)
    energy_optimized = df["energy_consumed"].iloc[0]
    energy_non_optimized = df["energy_consumed"].iloc[1]

    print(f"Emissions without optimization: {energy_non_optimized} KWh")
    print(f"Emissions with optimization: {energy_optimized} KWh")

    percentage_increase_energy = (energy_non_optimized - energy_optimized)/(energy_non_optimized + energy_optimized) * 100
    percentage_increase_rounded_energy = math.floor(percentage_increase_energy * 1000) / 1000
    
    st.session_state.processed_prompt = processing_results["processed_prompt"]
    st.session_state.original_length = processing_results["original_length"]
    st.session_state.processed_length = processing_results["processed_length"]
    st.session_state.reduction_percent = processing_results["reduction_percent"]
    st.session_state.ai_response = "Yes."
    st.session_state.energy_saving = percentage_increase_rounded_energy


def main():
    st.set_page_config(
        page_title="AI Prompt Processor Prototype",
        layout="centered",
    )

    st.title("🧠 Efficient AI Pipeline Demo")
    st.markdown("This application demonstrates a tiered architecture for **cost and energy efficiency** using custom classification and summarization.")

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

    st.subheader("1. Enter Your Raw Prompt")
    
    raw_prompt = st.text_area(
        "Input Prompt", 
        placeholder="e.g., Hi, I need a very detailed, multi-paragraph explanation of quantum computing, including its history and latest developments.", 
        height=250,
        key="current_raw_prompt_text_area"
    )
    
    if st.button("Run Optimized Pipeline", use_container_width=True, type="primary"):
        if not raw_prompt:
            st.error("Please enter a prompt to start the pipeline.")
        else:
            with st.spinner("Analyzing..."):
                run_processing_pipeline(raw_prompt)
    
    st.divider()
    
    if st.session_state.ai_response:
        
        st.subheader("2. Pipeline Status and Efficiency")
        
        col1, col2 = st.columns(2)
        
                
        with col1:
            st.metric("Reduction (%)", f"{st.session_state.reduction_percent}%")

        with col2:
            if st.session_state.is_greeting:
                st.metric("Efficiency Gain", "Max Savings")
            else:
                st.metric("Efficiency Gain", f"Saved {st.session_state.energy_saving}% KWh")

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
            <div style="padding: 20px; border-radius: 10px; border-left: 5px solid #10B981;">
                {st.session_state.ai_response}
            </div>
            """, 
            unsafe_allow_html=True
        )


if __name__ == '__main__':
    main()