import tensorflow as tf

def image_preprocess(image, new_size=(105, 84)):
    # Convert to grayscale, resize and normalize the image
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, new_size)
    image = image / 255.
    return image