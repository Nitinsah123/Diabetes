import customtkinter as ctk
from tkinter import messagebox
import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load & train model (same as before)
def load_model():
    data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", header=None)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, scaler

model, scaler = load_model()

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("🩺 Diabetes Prediction App")
app.geometry("850x700")

header = ctk.CTkLabel(app, text="Diabetes Prediction", font=("Helvetica", 28, "bold"), text_color="#1abc9c")
header.pack(pady=10)

subtitle = ctk.CTkLabel(app, text="Check if you're at risk and get personalized advice", font=("Helvetica", 16))
subtitle.pack(pady=5)

result_label = ctk.CTkLabel(app, text="", font=("Helvetica", 16, "bold"), text_color="#c0392b", wraplength=750, justify="left")
result_label.pack(pady=10)

form_frame = ctk.CTkFrame(app)
form_frame.pack(pady=20)

fields = [
    ("Pregnancies", "No. of Pregnancies"),
    ("Glucose", "Glucose level"),
    ("Blood Pressure", "Blood Pressure"),
    ("Skin Thickness", "Skin Thickness"),
    ("Insulin", "Insulin level"),
    ("BMI", "Body Mass Index"),
    ("Diabetes Pedigree", "Diabetes Pedigree Function"),
    ("Age", "Age")
]

entries = {}

# Create gender dropdown and label
def on_gender_change(choice):
    if choice == "Male":
        # Hide pregnancies label and entry completely
        preg_label.grid_remove()
        preg_entry.grid_remove()
        entries["Pregnancies"].delete(0, 'end')
        entries["Pregnancies"].insert(0, "0")  # preset 0 pregnancies for males
    else:
        # Show pregnancies label and entry
        preg_label.grid()
        preg_entry.grid()
        entries["Pregnancies"].delete(0, 'end')

gender_label = ctk.CTkLabel(form_frame, text="Gender", font=("Helvetica", 12, "bold"))
gender_label.grid(row=0, column=0, padx=20, pady=(10, 0), sticky="w")

gender_option = ctk.CTkOptionMenu(form_frame, values=["Male", "Female"], command=on_gender_change)
gender_option.grid(row=1, column=0, padx=20, pady=5)
gender_option.set("Male")

# Create input fields, but save pregnancies widgets separately for hide/show
for i, (label_text, placeholder) in enumerate(fields):
    row = (i + 1) // 3 + 1
    col = (i + 1) % 3
    label = ctk.CTkLabel(form_frame, text=label_text, font=("Helvetica", 12, "bold"))
    entry = ctk.CTkEntry(form_frame, placeholder_text=placeholder, width=200)
    
    if label_text == "Pregnancies":
        preg_label = label
        preg_entry = entry
    
    label.grid(row=row*2, column=col, padx=20, pady=(10, 0), sticky="w")
    entry.grid(row=row*2+1, column=col, padx=20, pady=5)
    entries[label_text] = entry

# Call once to set initial state of pregnancies field based on default gender
on_gender_change(gender_option.get())

def predict():
    try:
        gender = gender_option.get()
        values = []
        for label, _ in fields:
            val = entries[label].get()
            if val.strip() == "":
                raise ValueError("Missing input")
            values.append(float(val))

        input_data = scaler.transform([values])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        causes = []
        suggestions = []

        if values[1] > 140:
            causes.append("High Glucose Level")
            suggestions.append("Reduce sugar intake and monitor glucose levels.")
        if values[2] > 90:
            causes.append("High Blood Pressure")
            suggestions.append("Reduce salt intake and stay active.")
        if values[5] > 30:
            causes.append("High BMI")
            suggestions.append("Adopt a calorie-deficit diet and regular exercise.")
        if values[6] > 0.5:
            causes.append("Genetic Predisposition")
            suggestions.append("Consult a doctor for hereditary risk management.")
        if values[7] > 50:
            causes.append("Older Age")
            suggestions.append("Schedule regular health screenings.")

        if gender == "Male":
            suggestions.append("Avoid alcohol and smoking to improve insulin sensitivity.")
        else:
            suggestions.append("Balance hormones and manage weight, especially around menopause.")

        general_solutions = [
            "Engage in 30 mins of physical activity daily.",
            "Avoid processed food and drink more water.",
            "Get 7-8 hours of sleep every night.",
            "Add more fiber-rich vegetables and fruits to your meals.",
        ]

        selected = random.sample(suggestions, min(2, len(suggestions))) + random.sample(general_solutions, 2)

        result = (
            f"{'⚠️ You are likely to have Diabetes.' if prediction == 1 else '✅ You are unlikely to have Diabetes.'}\n\n"
            f"👤 Gender: {gender}\n"
            f"🔍 Identified Causes: {', '.join(causes) if causes else 'None'}\n"
            f"🧠 Probability Score: {probability:.2f}\n\n"
            f"💡 Personalized Advice:\n- " + "\n- ".join(selected)
        )

        result_label.configure(text=result)

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values in all fields.")

predict_btn = ctk.CTkButton(app, text="Analyze My Risk", command=predict,
                            fg_color="#1abc9c", text_color="white", hover_color="#16a085",
                            font=("Helvetica", 16), width=240)
predict_btn.pack(pady=30)

def toggle_mode():
    current = ctk.get_appearance_mode()
    ctk.set_appearance_mode("dark" if current == "Light" else "light")

toggle_switch = ctk.CTkSwitch(app, text="Dark Mode", command=toggle_mode)
toggle_switch.pack(pady=10)

app.mainloop()
