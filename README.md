# 🍏 Food Healthiness Prediction System (ML + Flask)

A machine learning-powered web application that predicts the **healthiness score (0–100)** of food items based on nutritional values.



## 🚀 Live Demo

👉 https://food-healthiness-predictor.onrender.com/



## 📌 Project Overview

This project uses the **USDA Food Composition Dataset** to evaluate food healthiness based on nutrients like protein, fiber, sugar, sodium, and fats.

It converts complex nutritional data into a simple, interpretable **health score**:

* 🟥 **Unhealthy** (0–33)
* 🟨 **Moderate** (34–66)
* 🟩 **Healthy** (67–100)



## 🧠 Machine Learning Details

* 📊 Dataset: USDA Food Composition Dataset (~8600 foods, 40+ nutrients)
* ⚙️ Features Selected: 15 key nutritional attributes
* 🤖 Model Used: Gradient Boosting Regressor
* 📈 Performance:

  * **R² Score:** 0.982
  * **RMSE:** 0.84
  * **MAE:** 0.28



## 🌐 Web Application Features

* 🔍 Search food items from dataset
* 📊 Predict healthiness score instantly
* 🍽️ Combined mode (multiple foods)
* 🎯 Clean and simple UI using Flask
* ⚡ Fast and accurate predictions



## 🛠️ Tech Stack

* Python
* Flask
* Scikit-learn
* Pandas & NumPy
* Joblib
* HTML/CSS



## 📂 Project Structure

```
.
├── food_selector_app.py
├── models/
│   ├── best_gradient_boosting.joblib
│   ├── scaler.joblib
│   └── selected_features.joblib
├── requirements.txt
├── Procfile
├── cleaned_food_dataset.xlsx
└── notebook/
```



## ⚙️ How to Run Locally

```bash
git clone https://github.com/your-username/food-healthiness-predictor.git
cd food-healthiness-predictor-ml
pip install -r requirements.txt
python food_selector_app.py
```

Open browser:

```
http://127.0.0.1:5000
```



## 🚀 Deployment

This project is deployed using **Render**.



## 📸 Preview

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/1bbb9220-1726-4ccc-a22e-bc57e61a8130" />
---
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/00110663-1fd3-450e-89f5-1f0c7d85b3c1" />
---
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/c1d7e298-0f5f-468c-ae60-2fec76f157d9" />
---
<img width="1999" height="1079" alt="image" src="https://github.com/user-attachments/assets/af314e3e-f1cf-4ba8-ad59-417c7fd25749" />
---
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/99d2ea2d-2023-464e-8018-2ff3297955c2" />




## 💡 Future Improvements

* Personalized diet recommendations
* Mobile-friendly UI
* Integration with food APIs
* Barcode scanning for packaged foods



## 👨‍💻 Author

**Utsav Ratpiya**
AI/ML Enthusiast | Data & Tech Explorer



## ⭐ If you like this project

Give it a ⭐ on GitHub and connect with me!
