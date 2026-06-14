# 🧠 LuminaNeuro

### AI-Powered Autonomous Triage for Structural MRI Scans

**Built for Code-Storm 26**

## 📌 The Problem

Emergency rooms and radiology departments capture thousands of MRI scans daily, creating a fatal bottleneck: humans have to read them sequentially. When a patient arrives with a fast-growing Glioblastoma, waiting days in a standard queue for a radiologist to review their file can be the difference between life and death. The current system is reactive.

## 🚀 Our Solution

**LuminaNeuro** is an autonomous triage system designed to act as a hyper-intelligent filter. It does not replace doctors; it ensures they see the most critical patients first.

By running incoming MRI matrices through a custom-trained deep learning model, LuminaNeuro categorizes structural anomalies in milliseconds. If a severe tumor is detected, the system bypasses the standard queue, visually alerts the medical team, and generates an immediate diagnostic dossier.

---

## ✨ Key Features (The "Flex" List)

* **⚡ Deep Learning Core:** Powered by a fine-tuned Convolutional Neural Network (EfficientNet architecture) that classifies scans into four categories (Glioma, Meningioma, Pituitary, or Nominal) with extremely high confidence.
* **🔴 Dynamic UI Engine:** A dependency-free, vanilla JavaScript frontend that reads the AI's output and dynamically recolors the entire DOM in real-time. Healthy scans maintain a clinical Teal, while detected anomalies trigger a Crimson Red critical alert state.
* **🎯 CSS-Rendered Thermal Heatmap:** If a tumor is detected, the system dynamically generates a targeted, non-destructive CSS-blended thermal/radar overlay directly on the uploaded MRI to highlight potential areas of interest.
* **📄 Export to EMR:** A built-in, one-click PDF generation tool that converts the web dashboard into a standardized medical report ready for the patient's Electronic Medical Record (EMR).
* **🛠️ Developer Auto-Fill:** A built-in dev toggle that instantly populates the matrix with mock data for rapid presentation and testing.

---

## 🏗️ The Tech Stack

### Backend

* **Python 3:** Core server logic.
* **Flask:** Lightweight API framework routing the frontend UI to the deep learning model.
* **TensorFlow / Keras:** Handles the image preprocessing, tensor normalization, and model prediction.
* **Pillow (PIL) & NumPy:** In-memory image processing and array conversion.

### Frontend

* **HTML5 & CSS3:** Custom-built "Glassmorphism" UI, cinematic CSS animations, and responsive flex/grid layouts. No CSS frameworks (like Bootstrap or Tailwind) were used.
* **Vanilla JavaScript (ES6):** Handles asynchronous `fetch` requests, `FormData` construction, `localStorage` data bridging, and DOM manipulation.

---

## ⚙️ Installation & Local Setup

To run LuminaNeuro locally on your machine:

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/LuminaNeuro.git
cd LuminaNeuro

```

**2. Install dependencies**

```bash
pip install -r requirements.txt

```

*(Requires: `flask`, `tensorflow`, `numpy`, `pillow`)*

**3. Add the AI Model**
Ensure your trained model file (`best_brain_tumor_model.keras`) is placed inside the `saved_models` directory:
`LuminaNeuro/saved_models/best_brain_tumor_model.keras`

**4. Start the Application**

```bash
python app.py

```

**5. Launch the UI**
Open your web browser and navigate to:
`http://127.0.0.1:5000`

---

## 🏥 User Workflow

1. **Ingestion Protocol:** The medical professional enters the Subject ID, Patient Name, DOB, and Gender on the home screen (`index.html`).
2. **Matrix Upload:** The user drags and drops the patient's `.jpg`, `.png`, or `.dcm` structural MRI scan.
3. **Initialization:** The backend processes the tensor array and returns the classification and confidence score.
4. **Diagnostic Dossier:** The user is seamlessly routed to the Unified Result Page (`result.html`). The UI assesses the risk level and dynamically formats the environment to either a Safe (Teal) or Critical (Red) state, displaying the scan, the AI heatmap, and actionable clinical next steps.

---

### 🏆 Built for Code-Storm 26

* **Team:** [Your Team Name]
* **Category:** Healthcare & Artificial Intelligence (AI)
