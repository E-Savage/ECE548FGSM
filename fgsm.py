import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False


# loading the class names and the imagenet class index
pretrained_model = tf.keras.applications.MobileNetV2(include_top=True,
                                                    weights='imagenet')

pretrained_model.trainable = False

# imageNet labels

decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

# function to help with the preprocessing of the image
def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = image[None, ...]
    return image

# helper function to extract labels from probability vector
def get_imagenet_label(probs):
    return decode_predictions(probs, top=1)[0][0]


# pulling original image 

image_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')


# print(tf.__version__)
# print("Hello, World!")