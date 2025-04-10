import gradio as gr
import modules.scripts as scripts
from modules import script_callbacks
from modules.processing import process_images, Processed
from modules.shared import opts, cmd_opts, state
import torch
from transformers import pipeline
import os
import sys
import json
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse

# Check if the required libraries are installed
try:
    from transformers import pipeline
except ImportError:
    print("Transformers library not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
    from transformers import pipeline

class TIPOTextGenerator:
    def __init__(self):
        self.pipe = None
        self.model_loaded = False
        self.model_name = "KBlueLeaf/TIPO-200M-ft2"
    
    def load_model(self):
        if not self.model_loaded:
            try:
                print(f"Loading TIPO model: {self.model_name}")
                self.pipe = pipeline("text-generation", model=self.model_name)
                self.model_loaded = True
                print("TIPO model loaded successfully!")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        return True
    
    def generate_text(self, prompt, max_length=200, temperature=0.7, top_p=0.9, num_return_sequences=1):
        if not self.model_loaded and not self.load_model():
            return "Failed to load the model."
        
        try:
            outputs = self.pipe(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True
            )
            
            generated_texts = [output['generated_text'] for output in outputs]
            return generated_texts[0] if num_return_sequences == 1 else generated_texts
        
        except Exception as e:
            return f"Error generating text: {e}"

# Global instance of the text generator
text_generator = TIPOTextGenerator()

class TIPOScript(scripts.Script):
    def title(self):
        return "TIPO Text Generator"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("TIPO Text Generator Settings", open=False):
                enabled = gr.Checkbox(label="Enable TIPO Text Enhancement", value=False)
                load_model = gr.Button(value="Load TIPO Model")
                model_status = gr.Textbox(label="Model Status", value="Not loaded", interactive=False)
                
                temperature = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, label="Temperature", value=0.7)
                top_p = gr.Slider(minimum=0.1, maximum=1.0, step=0.05, label="Top P", value=0.9)
                max_length = gr.Slider(minimum=50, maximum=500, step=10, label="Max Output Length", value=200)
                
                with gr.Row():
                    input_prompt = gr.Textbox(label="Input Prompt for Enhancement", lines=2)
                    generate_button = gr.Button(value="Generate & Insert")
                    
                enhanced_prompt = gr.Textbox(label="Enhanced Prompt", lines=3, interactive=False)
        
        def update_status():
            success = text_generator.load_model()
            return "Model loaded successfully!" if success else "Failed to load model"
        
        load_model.click(
            fn=update_status,
            outputs=[model_status]
        )
        
        def generate_enhanced_prompt(prompt, max_len, temp, p):
            if not prompt:
                return "Please provide an input prompt", prompt
            
            enhanced = text_generator.generate_text(
                prompt=prompt,
                max_length=max_len,
                temperature=temp,
                top_p=p
            )
            
            return enhanced, enhanced
        
        generate_button.click(
            fn=generate_enhanced_prompt,
            inputs=[input_prompt, max_length, temperature, top_p],
            outputs=[enhanced_prompt, input_prompt]
        )
        
        return [enabled, input_prompt, enhanced_prompt, temperature, top_p, max_length]

    def process(self, p, enabled, input_prompt, enhanced_prompt, temperature, top_p, max_length):
        if not enabled:
            return
            
        if enhanced_prompt and p.prompt:
            # Replace the original prompt with the enhanced one
            p.prompt = enhanced_prompt
            print(f"TIPO: Prompt enhanced to: {p.prompt}")

# Add the extension to the WebUI
def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as tipo_interface:
        with gr.Row():
            gr.Markdown("# TIPO Text Generator")
        
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(label="Input Prompt", lines=3, placeholder="Enter your prompt here...")
                with gr.Row():
                    temperature = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, label="Temperature", value=0.7)
                    top_p = gr.Slider(minimum=0.1, maximum=1.0, step=0.05, label="Top P", value=0.9)
                
                max_length = gr.Slider(minimum=50, maximum=500, step=10, label="Max Length", value=200)
                generate_btn = gr.Button(value="Generate Text")
            
            with gr.Column():
                output_text = gr.Textbox(label="Generated Text", lines=10)
                copy_to_prompt = gr.Button(value="Copy to Prompt")
        
        def generate_text(prompt, max_len, temp, p):
            if not prompt:
                return "Please provide an input prompt"
            
            # Make sure the model is loaded
            if not text_generator.model_loaded:
                text_generator.load_model()
                
            enhanced = text_generator.generate_text(
                prompt=prompt,
                max_length=max_len,
                temperature=temp,
                top_p=p
            )
            
            return enhanced
        
        generate_btn.click(
            fn=generate_text,
            inputs=[prompt_input, max_length, temperature, top_p],
            outputs=[output_text]
        )
        
        # This function would interact with the txt2img or img2img tabs
        def copy_to_txt2img():
            # This would need JavaScript integration to work with the main UI
            return
        
        copy_to_prompt.click(fn=copy_to_txt2img)
    
    return [(tipo_interface, "TIPO Generator", "tipo_generator")]

# Register the callbacks
script_callbacks.on_ui_tabs(on_ui_tabs)

# API endpoints for the TIPO text generator
def setup_api(app: FastAPI):
    @app.post("/tipo/generate")
    async def generate_tipo_text(
        prompt: str = Body(..., description="The input prompt to enhance"),
        max_length: int = Body(200, description="Maximum length of generated text"),
        temperature: float = Body(0.7, description="Temperature for text generation"),
        top_p: float = Body(0.9, description="Top-p sampling value"),
        num_return_sequences: int = Body(1, description="Number of sequences to return")
    ):
        # Ensure model is loaded
        if not text_generator.model_loaded:
            success = text_generator.load_model()
            if not success:
                return JSONResponse(
                    status_code=500,
                    content={"error": "Failed to load TIPO model"}
                )
        
        try:
            generated_text = text_generator.generate_text(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences
            )
            
            return JSONResponse(
                content={
                    "status": "success",
                    "input_prompt": prompt,
                    "generated_text": generated_text,
                    "parameters": {
                        "max_length": max_length,
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_return_sequences": num_return_sequences
                    }
                }
            )
        
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )
    
    @app.get("/tipo/status")
    async def get_tipo_status():
        return JSONResponse(
            content={
                "model_loaded": text_generator.model_loaded,
                "model_name": text_generator.model_name
            }
        )
    
    print("TIPO API endpoints registered: /tipo/generate and /tipo/status")
    return app

# Initialization function that will be called by the WebUI
def on_app_started(_, app):
    setup_api(app)

script_callbacks.on_app_started(on_app_started)
