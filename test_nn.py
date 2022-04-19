import tensorflow as tf
import cv2

def preprocess_image(path, shape):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, shape)
    img = img / 255.
    return img

def run():
    img_name = "banana"
    labels = ['clock', 'sun', 'pants', 'apple', 'tree', 'cloud', 'bridge', 'umbrella', 'crown',
              'snowman', 'computer', 'telephone', 'circle', 'rainbow', 'banana']
    path = f"./imgs/crown.jpg"
    shape = (28, 28)

    model = tf.keras.models.load_model("draw_model.h5")
    img = preprocess_image(path=path, shape=shape)
    preds = model.predict(tf.expand_dims(img, axis=0))
    num = max(preds[0])
    idx = list(preds[0]).index(num)
    print(f"Real class: {img_name}")
    print(f"Predicted: {labels[idx]}")


run()
