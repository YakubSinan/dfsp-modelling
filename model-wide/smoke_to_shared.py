# smoke_to_shared.py
from ultralytics import YOLO
from shared.schema import LesionPrediction, ModelOutput
from shared.validation import validate_model_output

# 1. Das YOLO modell 
model = YOLO("runs/detect/train/weights/best.pt")
result = model.predict("model-wide/dfsp_001.jpg", device="cpu")[0]

predictions = []

# 2. loops through the detected boxes and converts them to the shared schema
for i, box in enumerate(result.boxes, start=1):
    
    xc, yc, w, h = box.xywhn[0].tolist()   
    conf = float(box.conf[0])
    
    # 3 for every detected object, create a LesionPrediction instance and append it to the predictions list
    predictions.append(LesionPrediction(
        x=xc,
        y=yc,
        confidence=conf,
        lesion_id=f"smoke_lesion_{i}"
    ))

print(f"\n---> {len(predictions)}Detections are generated and translated!")

# 4. create a ModelOutput instance with the predictions
output = ModelOutput(
    image_id="dfsp_001",
    view="wide",
    predictions=predictions,
    patient_id="smoke_patient_001"
)

# 5. the ultimate validation step: check if the output matches the expected schema
validate_model_output(output)
print("---> Validation successful! The format is compatible.")