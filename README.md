# Deep Fake Detection Project

##  Overview

This project detects whether an input image is real or a deepfake using a trained deep learning model.

---

##  Features

* Image-based deepfake detection
* Simple Python implementation
* Easy to extend and integrate

---

##  Project Structure

DeepFake/
├── main.py        # Entry point of the application
├── model.py       # Model loading and prediction logic
├── test.jpg       # Sample input image
├── requirements.txt  # Dependencies
└── README.md      # Project documentation

---

##  Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Amith-143/Deep-Fake-Project.git
cd Deep-Fake-Project
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

##  Model Setup

The trained model file (`.h5`) is not included due to GitHub file size limits.

 Download the model from:
**[Add your Google Drive / Hugging Face link here]**

After downloading, place it inside:

```bash
DeepFake/
```

---

##  Run the Project

```bash
python main.py
```

---

##  Example

Input: `test.jpg`
Output: Prediction (Real / Fake)

---

##  Technologies Used

* Python
* TensorFlow / Keras (if used)
* NumPy
* OpenCV (if used)

---

##  Future Improvements

* Web interface using Streamlit or Flask
* Video deepfake detection
* Model optimization and accuracy improvements

---

## Contributing

Contributions are welcome. Feel free to fork the repo and submit a pull request.

---

##  License

This project is for educational purposes.

---

##  Author

Amith Marisa
GitHub: https://github.com/Amith-143
