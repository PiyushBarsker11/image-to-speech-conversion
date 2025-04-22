import os
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import GlobalAveragePooling2D
import numpy as np
import pickle
from gtts import gTTS
from playsound import playsound
import warnings
from keras.optimizers import Adam
from keras.initializers import Orthogonal

# Suppress warnings
warnings.filterwarnings("ignore")

# Print current working directory for debugging
print(f"Current working directory: {os.getcwd()}")

# Load the pre-trained captioning model with custom loss handling
try:
    caption_model = load_model("./model_weights/model_9.h5", custom_objects={'Orthogonal': Orthogonal()}, compile=False)
    print("Caption model loaded successfully!")
    caption_model.compile(optimizer=Adam(), loss='categorical_crossentropy')
except Exception as e:
    print(f"Error loading caption model: {e}")

# Load the ResNet50 model for image feature extraction with global average pooling
try:
    resnet_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(resnet_model.output)  # Add GlobalAveragePooling2D to reduce dimensions
    model_resnet = Model(resnet_model.input, x)
    print("ResNet50 model loaded successfully with GlobalAveragePooling!")
except Exception as e:
    print(f"Error loading ResNet50 model: {e}")

# Load word_to_idx and idx_to_word mappings
try:
    with open("./storage/word_to_idx.pkl", "rb") as w2i:
        word_to_idx = pickle.load(w2i)
    with open("./storage/idx_to_word.pkl", "rb") as i2w:
        idx_to_word = pickle.load(i2w)
    print("Dictionaries loaded successfully!")
except Exception as e:
    print(f"Error loading dictionaries: {e}")

# Maximum length of captions
max_len = 35

# Preprocess the image
def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Encode the image using ResNet50 with GlobalAveragePooling
def encode_image(img_path):
    img = preprocess_image(img_path)
    if img is None:
        return None
    try:
        feature_vector = model_resnet.predict(img)
        return feature_vector  # This will now return (1, 2048)
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

# Predict the caption based on the image
def predict_caption(photo):
    in_text = "startseq"
    for i in range(max_len):
        sequence = [word_to_idx.get(w, 0) for w in in_text.split()]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')
        y_pred = caption_model.predict([photo, sequence], verbose=0)
        y_pred = y_pred.argmax()
        word = idx_to_word.get(y_pred, None)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    final_caption = in_text.split()[1:]  # Skip "startseq"
    return ' '.join(final_caption[:-1])  # Skip "endseq"

# Text-to-Speech function
def speak_caption(caption):
    try:
        tts = gTTS(text=caption, lang='en')
        audio_path = os.path.join("static", "output_audio.mp3")  # Save in the static folder
        tts.save(audio_path)
        playsound(audio_path)
    except Exception as e:
        print(f"Error in speak_caption: {e}")

# Caption the image and speak the result
def caption_and_speak_image(img_path):
    # Your image encoding and caption generation logic
    photo = encode_image(img_path)
    if photo is None:
        print("Failed to encode image!")
        return "Error generating caption"  # In case of failure, return an error message
    
    caption = predict_caption(photo)
    print(f"Generated Caption: {caption}")
    
    if caption:
        speak_caption(caption)  # Function that converts caption to speech
        return caption  # Return the caption for display
    else:
        return "No caption generated. Please try another image."
