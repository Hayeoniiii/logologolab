from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
import base64
import io
from model.inference import load_model, generate_images

app = FastAPI()

@app.on_event("startup")
def startup_event():
    load_model()

@app.post("/generate-logo")
async def generate_logo(
    prompt: str = Body(...),
    style: str = Body(..., embed=True),         
    negative_prompt: str = Body("", embed=True),
    num_images: int = Body(1, embed=True)
):
    try:
        images = generate_images(prompt, style, negative_prompt, num_images)
        encoded_images = []

        for img in images:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            encoded_images.append(base64.b64encode(img_bytes).decode("utf-8"))

        return JSONResponse(content={"images": encoded_images})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
