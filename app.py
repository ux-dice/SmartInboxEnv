from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from inference import classify_email

app = FastAPI()

# ✅ REQUIRED FOR HACKATHON
@app.post("/openenv/reset")
def openenv_reset():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <body>
        <h1>Smart Inbox AI</h1>
    </body>
    </html>
    """


@app.get("/classify")
def classify(text: str):
    return classify_email(text)
