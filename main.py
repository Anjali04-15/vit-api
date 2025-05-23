from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from vit_model import VisionTransformer

# Same CLASS_NAMES as used in training
CLASS_NAMES = [
    'fresh_apple', 'fresh_banana', 'fresh_bitter_gourd', 'fresh_capsicum', 'fresh_orange', 'fresh_tomato',
    'stale_apple', 'stale_banana', 'stale_bitter_gourd', 'stale_capsicum', 'stale_orange', 'stale_tomato'
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict_from_model(image_tensor, model, device="cpu"):
    image_tensor = image_tensor.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        label = CLASS_NAMES[predicted_idx.item()]
        return {"label": label, "confidence": confidence.item()}

app = FastAPI()

# Allow CORS for all origins (for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def load_model_on_startup():
    # instantiate your model with matching architecture parameters
    model = VisionTransformer(
        img_size=224, 
        in_channels=3,
        patch_size=16,
        embedding_dims=768,
        num_transformer_layers=12,
        mlp_dropout=0.1,
        attn_dropout=0.0,
        mlp_size=3072,
        num_heads=12,
        num_classes=12
    )
    
    # load the full model weights (assuming saved full model, not just state_dict)
    model = torch.load("vit_weights.pt", map_location="cpu", weights_only=False)  
    model.eval()
    app.state.model = model
    
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        result = predict_from_model(image_tensor, app.state.model, device="cpu")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
