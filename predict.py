import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image, ImageTk
from torchvision import transforms, models
import tkinter as tk
from tkinter import filedialog

# -------- CONFIG --------
MODEL_PATH = "dr_results_v2/best_model.pth"
IMG_SIZE = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bias = np.array([1.0, 2.0, 1.0, 2.0, 1.0])

# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------- CLAHE --------
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# -------- MODEL --------
class EnsembleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = models.resnet50(weights="IMAGENET1K_V1")
        self.effnet = models.efficientnet_b3(weights="IMAGENET1K_V1")

        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.effnet = self.effnet.features

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Sequential(
            nn.Linear(2048 + 1536, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 5)
        )

    def forward(self, x):
        r = torch.flatten(self.resnet(x), 1)
        e = torch.flatten(self.pool(self.effnet(x)), 1)
        return self.fc(torch.cat((r, e), dim=1))

# -------- LOAD MODEL --------
model = EnsembleModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

print("Model loaded successfully")

# -------- GRAD-CAM++ --------
def grad_cam_pp(image, class_idx):
    gradients = []
    activations = []

    def f_hook(module, inp, out):
        activations.append(out)

    def b_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    layer = model.effnet[-1]
    layer.register_forward_hook(f_hook)
    layer.register_full_backward_hook(b_hook)

    output = model(image)
    model.zero_grad()
    output[0, class_idx].backward()

    grad = gradients[0][0].cpu().data.numpy()
    act = activations[0][0].cpu().data.numpy()

    weights = np.mean(grad, axis=(1,2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

    if cam.max() != 0:
        cam = cam / cam.max()

    return cam

# -------- PREDICTION --------
def run_prediction(path):

    img_cv = cv2.imread(path)
    img_cv = apply_clahe(img_cv)

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    img = transform(img_pil)

    batch = torch.stack([img, torch.flip(img, [2])]).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1).cpu().numpy()

    probs = probs.mean(axis=0)

    probs = probs * bias
    probs = probs / probs.sum()

    pred = np.argmax(probs)

    classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

    return pred, classes[pred], probs[pred], img_rgb, img.unsqueeze(0)

# -------- GUI --------
def open_image():
    file_path = filedialog.askopenfilename(
        title="Select Eye Image",
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )

    if file_path:
        result_label.config(text="Processing...")
        root.update()

        pred, label, conf, img_rgb, tensor = run_prediction(file_path)

        result_label.config(text=f"{label} ({conf*100:.2f}%)")

        # Show input image
        img = Image.open(file_path)
        img = img.resize((250, 250))
        img = ImageTk.PhotoImage(img)

        image_label.config(image=img)
        image_label.image = img

        # -------- Grad-CAM --------
        cam = grad_cam_pp(tensor.to(DEVICE), pred)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

        overlay = heatmap * 0.4 + img_resized
        overlay = np.uint8(overlay)

        overlay_img = Image.fromarray(overlay)
        overlay_img = overlay_img.resize((250, 250))
        overlay_img = ImageTk.PhotoImage(overlay_img)

        heatmap_label.config(image=overlay_img)
        heatmap_label.image = overlay_img

# -------- UI --------
root = tk.Tk()
root.title("Diabetic Retinopathy Detector")
root.geometry("500x600")

title = tk.Label(root, text="Upload Eye Image", font=("Arial", 16))
title.pack(pady=10)

upload_btn = tk.Button(root, text="Click to Upload Image", command=open_image)
upload_btn.pack(pady=20)

image_label = tk.Label(root)
image_label.pack()

heatmap_label = tk.Label(root)
heatmap_label.pack()

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=20)

root.mainloop()