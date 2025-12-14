import uvicorn
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration
THE_SETTINGS_DICT = {
    'model_name': 'gpt2-medium',
    'device': 'cpu', # Change to 'cuda' if you have a GPU
    'port': 8000
}

class InterpAgent:
    def __init__(self):
        print(f"Loading {THE_SETTINGS_DICT['model_name']}... (This may take a minute)")
        
        self.the_model = HookedTransformer.from_pretrained(
            THE_SETTINGS_DICT['model_name'], 
            device=THE_SETTINGS_DICT['device']
        )
        
        self.probes = {}
        self.steering_vectors = {}
        self.activations_for_viz = {}
        
        print(f"Model loaded!")

    def get_training_data(self):
        # A mix of facts and falsehoods for the probe to learn "Truthfulness"
        return [
            ("The capital of France is Paris", 1),
            ("The capital of Germany is Berlin", 1),
            ("The capital of Italy is Rome", 1),
            ("The capital of Spain is Madrid", 1),
            ("The capital of England is London", 1),
            ("The capital of Japan is Tokyo", 1),
            ("The capital of Australia is Canberra", 1),
            ("The capital of Canada is Ottawa", 1),
            ("The capital of France is London", 0),
            ("The capital of Germany is Paris", 0),
            ("The capital of Italy is Madrid", 0),
            ("The capital of Spain is Berlin", 0),
            ("The capital of England is Dublin", 0),
            ("The capital of Japan is Seoul", 0),
            ("The capital of Australia is Sydney", 0),
            ("The capital of Canada is Toronto", 0),
            ("Water is composed of hydrogen and oxygen", 1),
            ("The Earth orbits around the Sun", 1),
            ("Photosynthesis produces oxygen", 1),
            ("Humans have 46 chromosomes", 1),
            ("Water is composed of hydrogen and nitrogen", 0),
            ("The Sun orbits around the Earth", 0),
            ("Photosynthesis produces carbon dioxide", 0),
            ("Humans have 23 chromosomes", 0),
            ("Two plus two equals four", 1),
            ("Ten divided by two equals five", 1),
            ("A square has four sides", 1),
            ("Two plus two equals five", 0),
            ("Ten divided by two equals four", 0),
            ("A square has five sides", 0),
        ]

    def train_probe(self, layer):
        print(f"Training probe on layer {layer}...")
        
        data = self.get_training_data()
        prompts, labels = zip(*data)
        
        acts = []
        # Collect activations from the residual stream
        for i, txt in enumerate(prompts):
            if i % 10 == 0:
                print(f"   Processing {i+1}/{len(prompts)}...")
            
            # Run model with cache to get internal states
            _, cache = self.the_model.run_with_cache(txt, return_type=None)
            
            # Extract the activation of the last token at the specified layer
            # Shape: [batch, pos, d_model] -> [d_model]
            act = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :].cpu().detach().numpy()
            acts.append(act)
        
        X = np.array(acts)
        y = np.array(labels)
        
        # Train Logistic Regression (The "Probe")
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X, y)
        
        self.probes[layer] = clf
        self.steering_vectors[layer] = clf.coef_[0]
        self.activations_for_viz[layer] = (X, y)
        
        y_pred = clf.predict(X)
        acc = accuracy_score(y, y_pred)
        
        try:
            cm = confusion_matrix(y, y_pred).ravel().tolist()
        except:
            cm = [0, 0, 0, 0] # Fallback if matrix is weird
        
        print(f"Training complete. Accuracy: {acc*100:.1f}%")
        return acc, cm

    def get_confidence(self, text, layer):
        """Returns the probability that the statement is True according to the probe."""
        _, cache = self.the_model.run_with_cache(text, return_type=None)
        act = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :].cpu().detach().numpy().reshape(1, -1)
        return float(self.probes[layer].predict_proba(act)[0][1])

    def generate_steered(self, text, layer, coef):
        """Generates text while adding the 'truth vector' to activations."""
        vec = torch.tensor(self.steering_vectors[layer], dtype=torch.float32).to(THE_SETTINGS_DICT['device'])
        
        def hook_fn(resid, hook):
            # Add the steering vector * coefficient to the residual stream
            resid[:, :, :] += coef * vec
            return resid
        
        with self.the_model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)]):
            return self.the_model.generate(text, max_new_tokens=25, verbose=False, temperature=0.6, top_k=50)

    def generate_system2(self, text):
        """Chain of thought generation (slow thinking)."""
        prompt = f"Review the following claim.\nClaim: {text}\nAnalysis:"
        
        analysis = self.the_model.generate(prompt, max_new_tokens=40, verbose=False, temperature=0.7)
        
        final = self.the_model.generate(prompt + analysis + "\nTrue or False:", max_new_tokens=10, verbose=False, temperature=0.5)
        
        return analysis, final

    def get_pca_data(self, layer):
        """Reduces activations to 3D for visualization."""
        if layer not in self.activations_for_viz:
            return []
        
        X, y = self.activations_for_viz[layer]
        # Use PCA to project high-dim activations to 3D
        pca = PCA(n_components=3)
        res = pca.fit_transform(X)
        
        return [{"x": float(r[0]), "y": float(r[1]), "z": float(r[2]), "label": int(l)} for r, l in zip(res, y)]

# --- FastAPI Setup ---
app = FastAPI()
agent = InterpAgent()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainRequest(BaseModel):
    layer: int

class InferenceRequest(BaseModel):
    text: str
    layer: int
    steering_coef: float

@app.get("/")
def read_root():
    # Ensure templates directory exists
    if not os.path.exists("templates/index.html"):
        return HTMLResponse("<h1>Error: templates/index.html not found.</h1><p>Please save the HTML file provided in the response.</p>")
        
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/train")
def train(req: TrainRequest):
    acc, cm = agent.train_probe(req.layer)
    return {"accuracy": acc, "pca": agent.get_pca_data(req.layer), "confusion": cm}

@app.post("/infer")
def infer(req: InferenceRequest):
    if req.layer not in agent.probes:
        # Auto-train if not trained yet
        agent.train_probe(req.layer)
        
    conf = agent.get_confidence(req.text, req.layer)
    
    output = ""
    system_used = "System 1"
    trace = ""
    
    # Logic Router
    if req.steering_coef != 0:
        # If steering is active, force System 1 with modification
        output = agent.generate_steered(req.text, req.layer, req.steering_coef)
        system_used = "Steered"
    elif conf > 0.65:
        # If model is confident, use fast System 1
        output = agent.the_model.generate(req.text, max_new_tokens=25, verbose=False, temperature=0.6)
        system_used = "System 1 (Direct)"
    else:
        # If model is uncertain, use slow System 2 (Chain of Thought)
        trace, output = agent.generate_system2(req.text)
        system_used = "System 2 (CoT)"
        
    return {"confidence": conf, "output": output, "system": system_used, "trace": trace}

if __name__ == "__main__":
    print(f"Starting server on http://localhost:{THE_SETTINGS_DICT['port']}")
    uvicorn.run(app, host="0.0.0.0", port=THE_SETTINGS_DICT['port'])
