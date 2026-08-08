def pixel_to_normalized(x, y, image_width, image_height):
    x_normalized = x / image_width
    y_normalized = y / image_height

    return x_normalized, y_normalized


def normalized_to_pixel(x, y, image_width, image_height):
    x_pixel = round(x * image_width)
    y_pixel = round(y * image_height)

    return x_pixel, y_pixel