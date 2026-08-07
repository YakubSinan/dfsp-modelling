from shared.schema import LesionPrediction, ModelOutput
from shared.coordinates import pixel_to_normalized
from shared.validation import validate_model_output


x, y = pixel_to_normalized(
    x=960,
    y=540,
    image_width=1920,
    image_height=1080
)

prediction = LesionPrediction(
    x=x,
    y=y,
  confidence=0.90,
    lesion_id="lesion_001"
)

output = ModelOutput(
    image_id="p001_back_wide",
    view="wide",
    predictions=[prediction],
    patient_id="p001"
)

validate_model_output(output)

print("Shared interface works correctly.")
print(output)