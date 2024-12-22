import numpy as np
import cv2
import yaml
import time
import cProfile


with open("data.yaml", "r") as file:
    data = yaml.safe_load(file)
    classes = data['names']
cap = cv2.VideoCapture(1)
net = cv2.dnn.readNetFromONNX("best.onnx")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

while True:
    start_time = time.time()

    _, img = cap.read()
    capture_time = time.time()

    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=[640, 640], mean=[0, 0, 0], swapRB=True, crop=False)
    net.setInput(blob)
    preprocess_time = time.time()
    detections = net.forward()[0]
    inference_time = time.time()


    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]

    img_width, img_height = img.shape[1], img.shape[0]
    x_scale = img_width/640
    y_scale = img_height/640

    for i in range(rows):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.2:
            classes_score = row[5:]
            ind = np.argmax(classes_score)
            if classes_score[ind] > 0.2:
                classes_ids.append(ind)
                confidences.append(confidence)
                cx, cy, w, h = row[:4]
                x1 = int((cx-w/2)*x_scale)
                y1 = int((cy-h/2)*y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                box = np.array([x1, y1, width, height])
                boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.2)

    # Pastikan indices tidak kosong
    if len(indices) > 0:
        for i in indices.flatten():
            x1, y1, w, h = boxes[i]
            label = classes[classes_ids[i]]
            conf = confidences[i]
            text = f"{label} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
            cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
    postprocess_time = time.time()

    print(f"Capture Time: {capture_time - start_time:.3f}s")
    print(f"Preprocessing Time: {preprocess_time - capture_time:.3f}s")
    print(f"Inference Time: {inference_time - preprocess_time:.3f}s")
    print(f"Postprocessing Time: {postprocess_time - inference_time:.3f}s")
    print(f"Total Loop Time: {postprocess_time - start_time:.3f}s")
    cv2.imshow("Deteksi Objek", img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break