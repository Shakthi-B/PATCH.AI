import onnxruntime
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score


ort_session = onnxruntime.InferenceSession("your_model.onnx")


def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")
    #image = np.array(image.resize((input_width, input_height)))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)  
    return image


test_images = ["test_image1.jpg", "test_image2.jpg", ...]
test_labels = [0, 1, ...] 


predictions = []
for image_path in test_images:
    preprocessed_image = preprocess_image(image_path)
    ort_inputs = {ort_session.get_inputs()[0].name: preprocessed_image}
    ort_outs = ort_session.run(None, ort_inputs)
    predictions.append(np.argmax(ort_outs[0]))


accuracy = accuracy_score(test_labels, predictions)
print("Accuracy:", accuracy)