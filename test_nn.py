import tensorflow as tf
import cv2

def preprocess_image(path, shape):
    img = cv2.imread(path)
    img = cv2.resize(img, shape)
    img = img / 255
    return img

def run():


    img_name = "computer"
    labels = ['snowman', 'apple', 'telephone', 'tree', 'circle', 'banana', 'crown', 'pants', 'sun', 'clock',
              'rainbow', 'umbrella', 'cloud', 'bridge', 'computer']
    path = f"./imgs/computer.png"
    shape = (28, 28)

    model = tf.keras.models.load_model("draw_model.h5")
    img = preprocess_image(path=path, shape=shape)
    test = cv2.bitwise_not(img)
    # cv2.imshow("t", test)
    # cv2.waitKey(0)
    preds = model.predict(tf.expand_dims(img, axis=0))
    num = max(preds[0])
    idx = list(preds[0]).index(num)
    print(labels)
    print(f"Real class: {img_name}")
    print(f"Predicted: {labels[idx]}")


run()
