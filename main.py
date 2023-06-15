import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import onnxruntime as ort
from PIL import ImageTk, Image

# Load the ONNX model
sess = ort.InferenceSession('best.onnx')

# Tạo cửa sổ
root = tk.Tk()

# Định nghĩa kích thước cửa sổ
width = 300
height = 60
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{width}x{height}+{int((screen_width-width)/2)}+{int((screen_height-height)/2)}")

# Hàm xử lý khi nhấn nút "Open file"
def open_file():
    # Chọn tệp tin ảnh
    file_path = filedialog.askopenfilename()
    print("Selected file path:", file_path)

    # Đọc ảnh bằng OpenCV
    img = cv2.imread(file_path)

    img = cv2.resize(img, (640, 640))

    # Chuẩn hóa giá trị pixel về khoảng [0, 1]
    img = img / 255.0

    # Đưa ảnh vào model để dự đoán bounding box
    input_blob = np.expand_dims(img.transpose(2, 0, 1), axis=0).astype(np.float32)
    outputs = sess.run(None, {'images': input_blob})
    if len(outputs) > 0:
        boxes = outputs[0]
    if len(outputs) > 1:
        scores = outputs[1]
    else:
        scores = None
    if len(outputs) > 2:
        classes = outputs[2]

    # Vẽ bounding box trên ảnh
    threshold = 0.5  # Ngưỡng cho score của bounding box
    iou_threshold = 0.4  # Ngưỡng cho IoU giữa các bounding box
    for i in range(len(boxes)):
        for j in range(len(boxes[i])):
            if scores is not None and scores[i][j] > threshold:
                box = boxes[i][j]
                x1, y1, x2, y2 = box
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f"{classes[i][j]:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                for k in range(j+1, len(boxes[i])):
                    if scores[i][k] > threshold and cv2.dnn.NMSBoxes([box, boxes[i][k]], [scores[i][j], scores[i][k]], threshold, iou_threshold):
                        box_k = boxes[i][k]
                        x1_k, y1_k, x2_k, y2_k = box_k
                        cv2.rectangle(img, (int(x1_k), int(y1_k)), (int(x2_k), int(y2_k)), (0, 0, 255), 2)

    # Chuyển đổi ảnh numpy array sang định dạng Image của PIL
    img_pil = Image.fromarray((img * 255).astype(np.uint8))

    # Tạo một cửa sổ mới để hiển thị ảnh và kết quả dự đoán
    new_window = tk.Toplevel(root)

    # Tạo một widget Label để hiển thị ảnh trên cửa sổ mới
    label = tk.Label(new_window)
    label.pack()

    # Chuyển đổi ảnh sang định dạng PhotoImage của tkinter
    img_tk = ImageTk.PhotoImage(img_pil)

    # Hiển thị ảnh trên widget Label
    label.config(image=img_tk)
    label.image = img_tk

    # Hiển thị kết quả dự đoán bounding box
    if scores is not None:
        for i in range(len(boxes)):
            for j in range(len(boxes[i])):
                if scores[i][j] > threshold:
                    box = boxes[i][j]
                    x1, y1, x2, y2 = box
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(img, f"{classes[i][j]:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    for k in range(j+1, len(boxes[i])):
                        if scores[i][k] > threshold and cv2.dnn.NMSBoxes([box, boxes[i][k]], [scores[i][j], scores[i][k]], threshold, iou_threshold):
                            box_k = boxes[i][k]
                            x1_k, y1_k, x2_k, y2_k = box_k
                            cv2.rectangle(img, (int(x1_k), int(y1_k)), (int(x2_k), int(y2_k)), (0, 0, 255), 2)

        # Cập nhật lại ảnh sau khi vẽ bounding box
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        img_tk = ImageTk.PhotoImage(img_pil)
        label.config(image=img_tk)
        label.image = img_tk

# Tạo nút "Open file"
button = tk.Button(root, text="Open file", command=open_file)
button.pack()

# Hiển thị cửa sổ
root.mainloop()
