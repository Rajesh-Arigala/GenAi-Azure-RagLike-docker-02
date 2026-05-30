import os
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import PyPDF2
from io import BytesIO
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for Azure AI Inference client
client = None
phi4_deployment = None

def find_cred_json(start_path: str) -> str | None:
    """
    Recursively looks for 'cred.json' under start_path.
    Returns the first match or None if not found.
    """
    base = Path(start_path)
    print(f"🔎 Searching for cred.json under: {base.resolve()}")
    for candidate in base.rglob('cred.json'):
        print(f"✅ Found cred.json at: {candidate}")
        return str(candidate)
    return None

def initialize_azure_client():
    """Initialize the Azure AI Inference client"""
    global client, phi4_deployment
    
    try:
        # 1. Locate cred.json anywhere beneath the current directory
        cwd = os.getcwd()
        file_path = find_cred_json(cwd)
        if not file_path:
            raise FileNotFoundError("cred.json not found under the current directory")

        # 2. Load and parse the JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        
        # 3. Extract configuration
        endpoint = cfg.get("ENDPOINT", "")
        api_key = cfg.get("API_KEY", "")
        phi4_deployment = cfg.get("MODEL_DEPLOYMENT_NAME", "Phi-4")
        
        print(f"Endpoint:                    {endpoint}")
        print(f"Model Deployment Name:      {phi4_deployment}")

        # 4. Create the ChatCompletionsClient
        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
            api_version="2024-05-01-preview"
        )
        print("✅ ChatCompletionsClient created successfully!")
        return True

    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ JSON error in cred.json: {e}")
        return False
    except KeyError as e:
        print(f"❌ Missing key in config: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def chat_with_phi4_rag(user_question, retrieved_doc):
    """Simulate an RAG flow by appending retrieved context to the system prompt."""
    global client, phi4_deployment
    
    if not client:
        return "Error: Azure AI Inference client not initialized"
    
    system_prompt = (
        "You are Phi-4, a helpful fitness AI.\n"
        "We have some context from the user's knowledge base:\n"
        f"{retrieved_doc}\n"
        "Please use this context to help your answer. If the context doesn't help, say so.\n"
    )

    try:
        response = client.complete(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_question)
            ],
            model=phi4_deployment,
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error communicating with Azure AI Inference: {str(e)}"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Extract text from PDF
            pdf_text = extract_text_from_pdf(BytesIO(file.read()))
            
            return jsonify({
                'success': True,
                'text': pdf_text
            })
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/ask-question', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        pdf_text = data.get('context', '')
        question = data.get('question', '')
        
        if not pdf_text or not question:
            return jsonify({'error': 'Missing context or question'}), 400
        
        # Use Azure AI Inference for chat completion
        answer = chat_with_phi4_rag(question, pdf_text)
        
        return jsonify({
            'answer': answer,
            'success': True
        })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize Azure AI Inference client on startup
    if initialize_azure_client():
        print("🚀 Flask app starting with Azure AI Inference integration...")
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    else:
        print("❌ Failed to initialize Azure AI Inference client. Please check your cred.json file.")
        print("App will not start without proper Azure configuration.")