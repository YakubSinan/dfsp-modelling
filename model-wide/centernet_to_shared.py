
from shared.schema import LesionPrediction, ModelOutput
from shared.validation import validate_model_output


# we simulate the result from the 16x16 heatmap of the CenterNet.
# In the practice, here the brightest point (argmax) in the grid is searched.
# Let's assume the model finds the center of the lesion at grid position x=12, y=4.
grid_size = 16.0
predicted_grid_x = 12.0
predicted_grid_y = 4.0
predicted_confidence = 0.88

# 1. normalize the coordinates to [0, 1] range
normalized_x = predicted_grid_x / grid_size
normalized_y = predicted_grid_y / grid_size

print(f"Normalisierte Koordinaten: x={normalized_x}, y={normalized_y}")

# 2. in the shared schema, we create a LesionPrediction object
prediction = LesionPrediction(
    x=normalized_x,
    y=normalized_y,
    confidence=predicted_confidence,
    lesion_id="cn_smoke_001"
)

# 3. ModelOutput build (mit view="wide")
output = ModelOutput(
    image_id="centernet_test_img",
    view="wide",
    predictions=[prediction],
    patient_id="patient_company_002"
)

# 4. against the schema, we validate the output
validate_model_output(output)
print("---> CenterNet-Smoke-Test passed. the output is valid.")