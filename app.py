from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from inference import main as run_inference

app = FastAPI()

# ✅ THIS IS REQUIRED FOR HACKATHON
@app.post("/openenv/reset")
def openenv_reset():
    return {"status": "ok"}


# ✅ REQUIRED: inference trigger endpoint
@app.post("/openenv/run")
def openenv_run():
    run_inference()
    return {"status": "completed"}


# Optional UI (safe)
@app.get("/", response_class=HTMLResponse)
def home():
    return "<h1>Smart Inbox AI Running 🚀</h1>"
