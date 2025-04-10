import gradio as gr
import modules.scripts as scripts
from modules import script_callbacks
import os
import sys
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse

# Check if the required libraries are installed
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError:
    print("Required libraries not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Hide all CUDA devices to avoid mixed device errors
os.environ["CUDA_VISIBLE_DEVICES"] = ""

class TIPOTextGenerator:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.model_loaded = False
        self.model_name = "KBlueLeaf/DanTagGen-beta"
    
    def load_model(self):
        if not self.model_loaded:
            try:
                print(f"Loading TIPO model: {self.model_name} on CPU (simple version)")
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
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
            # Manual generation without pipeline
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Set generation parameters
            gen_kwargs = {
                "max_length": max_length,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "num_return_sequences": num_return_sequences,
            }
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **gen_kwargs)
            
            generated_texts = [
                self.tokenizer.decode(g, skip_special_tokens=True)
                for g in generated_ids
            ]
            
            return generated_texts[0] if num_return_sequences == 1 else generated_texts
        
        except Exception as e:
            print(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error generating text: {e}"

# Global instance of the text generator
text_generator = TIPOTextGenerator()

# Create a simple script for the WebUI
class TIPOScript(scripts.Script):
    def title(self):
        return "TIPO Text Generator (Simple)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("TIPO Text Generator (Simple)", open=False):
                enabled = gr.Checkbox(label="Enable TIPO Text Enhancement", value=False)
                load_model = gr.Button(value="Load TIPO Model")
                model_status = gr.Textbox(label="Model Status", value="Not loaded", interactive=False)
                
                with gr.Row():
                    input_prompt = gr.Textbox(label="Input Prompt for Enhancement", lines=2)
                    generate_button = gr.Button(value="Generate Text")
                    
                enhanced_prompt = gr.Textbox(label="Enhanced Prompt", lines=3, interactive=False)
        
        def update_status():
            success = text_generator.load_model()
            return "Model loaded successfully!" if success else "Failed to load model"
        
        load_model.click(
            fn=update_status,
            outputs=[model_status]
        )
        
        def generate_enhanced_prompt(prompt):
            if not prompt:
                return "Please provide an input prompt"
            
            enhanced = text_generator.generate_text(prompt=prompt)
            return enhanced
        
        generate_button.click(
            fn=generate_enhanced_prompt,
            inputs=[input_prompt],
            outputs=[enhanced_prompt]
        )
        
        return [enabled, input_prompt, enhanced_prompt]

    def process(self, p, enabled, input_prompt, enhanced_prompt):
        if not enabled or not enhanced_prompt:
            return
        
        # Replace the original prompt with the enhanced one
        p.prompt = enhanced_prompt
        print(f"TIPO: Prompt enhanced to: {p.prompt}")

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
