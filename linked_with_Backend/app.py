from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import base64
import io
from model.inference import load_model, generate_images
from fastapi import Body

app = FastAPI()

@app.on_event("startup")
def startup_event():
    load_model()


@app.post("/generate-logo")
async def generate_logo(
    prompt: str = Body(...),                 
    negative_prompt: str = Body(""),
    num_images: int = Body(1)
):
    try:
        images = generate_images(prompt, negative_prompt, num_images)
        encoded_images = []

        for img in images:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            encoded = base64.b64encode(img_bytes).decode("utf-8")
            encoded_images.append(encoded)

        return JSONResponse(content={"images": encoded_images})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
