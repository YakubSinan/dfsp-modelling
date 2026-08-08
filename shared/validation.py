from .schema import ModelOutput


def validate_model_output(output: ModelOutput):
    if output.view not in {"wide", "close"}:
        raise ValueError(
            f"view must be 'wide' or 'close', got '{output.view}'"
        )

    if not output.image_id:
        raise ValueError("image_id cannot be empty")

    for prediction in output.predictions:
        if not 0.0 <= prediction.x <= 1.0:
            raise ValueError(
                f"x must be between 0 and 1, got {prediction.x}"
            )

        if not 0.0 <= prediction.y <= 1.0:
            raise ValueError(
                f"y must be between 0 and 1, got {prediction.y}"
            )

        if not 0.0 <= prediction.confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0 and 1, got {prediction.confidence}"
            )

    return True