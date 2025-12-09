from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import pandas as pd
import sys
import os

# Add the root directory to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.prediction_pipeline import PredictPipeline, CustomData

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Initialize Pipeline ONCE to load artifacts and get metadata
pipeline = PredictPipeline()

# Extract Metadata for Dropdowns & Autofill
# 1. Center Metadata: center_id -> {city_code, region_code, center_type, op_area}
center_meta_df = pipeline.merged[['center_id', 'city_code', 'region_code', 'center_type', 'op_area']].drop_duplicates().set_index('center_id')
center_info = center_meta_df.to_dict('index')

# 2. Meal Metadata: meal_id -> {category, cuisine}
meal_meta_df = pipeline.merged[['meal_id', 'category', 'cuisine']].drop_duplicates().set_index('meal_id')
meal_info = meal_meta_df.to_dict('index')

# Lists for Dropdowns
center_ids = sorted(list(center_info.keys()))
meal_ids = sorted(list(meal_info.keys()))

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "center_ids": center_ids,
        "meal_ids": meal_ids,
        "center_info": center_info,
        "meal_info": meal_info
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    week: int = Form(...),
    center_id: int = Form(...),
    meal_id: int = Form(...),
    checkout_price: float = Form(...),
    base_price: float = Form(...),
    emailer_for_promotion: int = Form(...),
    homepage_featured: int = Form(...),
    city_code: int = Form(...),
    region_code: int = Form(...),
    center_type: str = Form(...),
    op_area: float = Form(...),
    category: str = Form(...),
    cuisine: str = Form(...)
):
    try:
        data = CustomData(
            week=week,
            center_id=center_id,
            meal_id=meal_id,
            checkout_price=checkout_price,
            base_price=base_price,
            emailer_for_promotion=emailer_for_promotion,
            homepage_featured=homepage_featured,
            city_code=city_code,
            region_code=region_code,
            center_type=center_type,
            op_area=op_area,
            category=category,
            cuisine=cuisine
        )
        
        pred_df = data.get_data_as_data_frame()
        print(f"Prediction DataFrame:\n{pred_df}")

        # pred_df = data.get_data_as_data_frame()
        # print(f"Prediction DataFrame:\n{pred_df}")

        # pipeline is already initialized globally
        results = pipeline.predict(pred_df)

        return templates.TemplateResponse("result.html", {"request": request, "result": round(results)})

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": str(e),
            "center_ids": center_ids,
            "meal_ids": meal_ids,
            "center_info": center_info,
            "meal_info": meal_info
        })

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
