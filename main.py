import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
import streamlit as st
import h5py 

import warnings
warnings.simplefilter('ignore')

# Loading the train & test data -
train = pd.read_csv('train2.csv')
test = pd.read_csv('test2.csv')

# Splitting the data into independent & dependent variables -
X_train, y_train = base.splitter(train, y_var='DISEASE')
X_test, y_test = base.splitter(test, y_var='DISEASE')

# Standardizing the data -
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = keras.Sequential([
    keras.layers.Dense(units=128, input_shape=(13,), activation='relu', kernel_regularizer=regularizers.l2(2.0)),
    keras.layers.BatchNormalization(axis=1),
    keras.layers.Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.BatchNormalization(axis=1),
    keras.layers.Dense(units=32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.BatchNormalization(axis=1),
    keras.layers.Dense(units=16, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.BatchNormalization(axis=1),
    keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))
])
adam = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, mode='min', verbose=1)
mc = tf.keras.callbacks.ModelCheckpoint(filepath='model.weights.h5', monitor='val_accuracy', save_best_only=True, save_weights_only=True)

print("X_train_scaled shape:", X_train_scaled.shape)  # Should be (samples, 13)
print("X_test_scaled shape:", X_test_scaled.shape)

hist = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), 
                 epochs=100, batch_size=32, callbacks=[es, mc], verbose=1)
_, train_acc = model.evaluate(X_train_scaled, y_train, batch_size=32, verbose=0)
_, test_acc = model.evaluate(X_test_scaled, y_test, batch_size=32, verbose=0)

print('Train Accuracy: {:.3f}'.format(train_acc))
print('Test Accuracy: {:.3f}'.format(test_acc))

y_pred_proba = model.predict(X_test_scaled, batch_size=32, verbose=0)

threshold = 0.50
y_pred_class = np.where(y_pred_proba > threshold, 1, 0)
 
# Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='test')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Accuracy Curve
plt.figure(figsize=(10, 5))
plt.plot(hist.history['accuracy'], label='train')
plt.plot(hist.history['val_accuracy'], label='test')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# ROC Curve
from sklearn.metrics import roc_auc_score, roc_curve

logit_roc_auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr, label='(area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import precision_recall_curve, confusion_matrix

# Precision-Recall Curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

min_length = min(len(thresholds), len(precisions) - 1, len(recalls) - 1)
thresholds = thresholds[:min_length]
precisions = precisions[:min_length]
recalls = recalls[:min_length]

plt.plot(thresholds, precisions, label='Precision')
plt.plot(thresholds, recalls, label='Recall')
plt.xlabel('Threshold Value')
plt.ylabel('Precision and Recall Value')
plt.legend()
plt.grid()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_class))

model.save('model.h5')

attrib_info = """
#### Fields:
    -age
    -sex
    -cp
    -trestbps
    -chol
    -fbs
    -restecg
    -thalach
    -exang
    -oldpeak
    -slope
"""

@st.cache_resource()#allow_output_mutation=True)


@st.cache_resource()#allow_output_mutation=True)
def load_model(model_file):
    model = tf.keras.models.load_model('model.h5')
    return model

def ann_app():
    # Add custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 30px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 30px;
            color: #1E88E5;
        }
        .stExpander {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .metric-header {
            color: #1E88E5;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            background-color: #f0f8ff;
            margin: 20px 0;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # Main header
    st.markdown('<p class="main-header">Heart Disease Prediction Model</p>', unsafe_allow_html=True)
    
# Streamlit application
@st.cache_resource
def load_keras_model():
    return tf.keras.models.load_model('model.h5')

loaded_model = load_keras_model()

def ann_app():
    st.markdown("<h1 style='text-align: center; color: #1E88E5;'>Heart Disease Prediction</h1>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📝 Input Parameters", "📊 Model Metrics"])

    with tab1:
        st.markdown("### Patient Information")
        
        # Collect user inputs
        AGE = st.number_input("Age", min_value=0, max_value=120, step=1)
        RESTING_BP = st.number_input("Resting Blood Pressure", min_value=0, max_value=300, step=1)
        SERUM_CHOLESTROL = st.number_input("Serum Cholesterol", min_value=0, max_value=1000, step=1)
        TRI_GLYCERIDE = st.number_input("Triglycerides", min_value=0, max_value=1000, step=1)
        LDL = st.number_input("LDL", min_value=0, max_value=300, step=1)
        HDL = st.number_input("HDL", min_value=0, max_value=100, step=1)
        FBS = st.number_input("Fasting Blood Sugar", min_value=0, max_value=500, step=1)
        GENDER = st.selectbox('Gender', options=["Female", "Male"], index=0)
        CHEST_PAIN = st.selectbox('Chest Pain', options=["No", "Yes"], index=0)
        RESTING_ECG = st.selectbox('Resting ECG', options=["Normal", "Abnormal"], index=0)
        TMT = st.selectbox('TMT (Treadmill Test)', options=["Normal", "Abnormal"], index=0)
        ECHO = st.number_input("Echo", min_value=0, max_value=100, step=1)
        MAX_HEART_RATE = st.number_input("Maximum Heart Rate", min_value=0, max_value=250, step=1)

        

        
        # Add a predict button
        # Add a predict button
        if st.button('Predict'):
            # Prepare input
            sample = np.array([AGE, GENDER == "Male", CHEST_PAIN == "Yes", RESTING_BP, SERUM_CHOLESTROL,
                            TRI_GLYCERIDE, LDL, HDL, FBS, RESTING_ECG == "Abnormal", MAX_HEART_RATE,
                            ECHO, TMT == "Abnormal"]).reshape(1, -1)
            sample_scaled = scaler.transform(sample)

            # Prediction
            prediction = loaded_model.predict(sample_scaled)[0][0]
            predicted_percentage = prediction * 100

            # Determine risk level with style
            if predicted_percentage < 40:
                risk_level = "🟢 Low Risk"
                advice = "Your heart health looks good! Maintain a healthy diet and stay active. 😊"
                color = "#4CAF50"  # Green
                st.markdown("### 🍽 Diet Preferences")
                st.markdown("""
                        *🎉 Low Risk & On a Diet*:
                        - ✅ *Maintain a balanced diet* — fruits, greens, healthy fats.
                        - 🍎 *Include colorful veggies, grains, and lean protein*.
                        - ✨ Keep up your great work — *your heart is smiling!* 😊
                        """)
            elif 40 <= predicted_percentage < 70:
                risk_level = "🟡 Moderate Risk"
                advice = "Consider lifestyle improvements like regular exercise and a heart-healthy diet. 🏃🍏"
                color = "#FFC107"  # Yellow
                st.markdown("### 🍽 Diet Preferences")
                st.markdown("""
                        *⚡ Moderate Risk & On a Diet*:
                        - 🍎 *Increase fiber intake* from fruits, veggies, whole grains.
                        - 🥩 Balance *lean protein* and *healthy fats* (avoid junk).
                        - 🏋 *Exercise regularly* to enhance results.
                        """)
            else:
                risk_level = "🔴 High Risk"
                advice = "It's recommended to consult a doctor for further evaluation and guidance. 🚑"
                color = "#F44336"  # Red
                st.markdown("### 🍽 Diet Preferences")
                st.markdown("""
                        *🚨 High Risk & On a Diet*:
                        - 🥗 Prioritize *whole grains, **lean proteins, **healthy fats*.
                        - ❌ *Cut out processed, fried foods, and sugar-heavy snacks*.
                        - 🩺 *Consult a dietitian* for an optimized heart-healthy plan.
                        """)

            # Display the styled result box
            st.markdown(
                f"""
                <div style="
                    border-radius: 12px;
                    background-color: #f5f5f5;
                    padding: 20px;
                    text-align: center;
                    box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
                ">
                    <h2 style="color: #333;">Prediction Result</h2>
                    <h1 style="font-size: 48px; color: {color};">{predicted_percentage:.2f}%</h1>
                    <h3 style="color: {color};">{risk_level}</h3>
                    <p style="color: #555;">{advice}</p>
                </div>
                """,
                unsafe_allow_html=True
            )



        with tab2:
            st.markdown("""
            ### Model Evaluation Metrics
            
            Explore the various metrics used to evaluate the model's performance:
            """)
            
            metrics = st.radio(
                "Select Metric to View:",
                ["ROC-AUC Curve", "Model Loss", "Model Accuracy", "Precision-Recall", "Confusion Matrix"],
                horizontal=True
        )
        
        # Create expandable sections for each plot
        if metrics == "ROC-AUC Curve":
            with st.expander("📈 ROC-AUC Curve", expanded=True):
                st.markdown("#### Receiver Operating Characteristic (ROC) Curve")
                logit_roc_auc = roc_auc_score(y_test, y_pred_proba)
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_class)
                
                fig = plt.figure(figsize=(10, 6))
                plt.plot(fpr, tpr, label=f'AUC = {logit_roc_auc:.2f}')
                plt.plot([0, 1], [0, 1], 'r--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                plt.grid(True)
                st.pyplot(fig)
                
        elif metrics == "Model Loss":
            with st.expander("📉 Model Loss Curve", expanded=True):
                st.markdown("#### Training and Validation Loss")
                fig = plt.figure(figsize=(10, 6))
                plt.plot(hist.history['loss'], label='Training Loss')
                plt.plot(hist.history['val_loss'], label='Validation Loss')
                plt.title('Model Loss Over Time')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend(loc='upper right')
                plt.grid(True)
                st.pyplot(fig)
                
        elif metrics == "Model Accuracy":
            with st.expander("📊 Model Accuracy Curve", expanded=True):
                st.markdown("#### Training and Validation Accuracy")
                fig = plt.figure(figsize=(10, 6))
                plt.plot(hist.history['accuracy'], label='Training Accuracy')
                plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
                plt.title('Model Accuracy Over Time')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.legend(loc='lower right')
                plt.grid(True)
                st.pyplot(fig)
                
        elif metrics == "Precision-Recall":
          with st.expander('Precision-Recall Plot'):
            st.subheader("Precision-Recall Plot")
            try:
              precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
            
              # Ensure all arrays have the same length
              min_length = min(len(thresholds), len(precisions) - 1, len(recalls) - 1)
              thresholds = thresholds[:min_length]
              precisions = precisions[:min_length]
              recalls = recalls[:min_length]

              fig4 = plt.figure(figsize=(10, 5))
              plt.plot(thresholds, precisions[:min_length], label='Precision')
              plt.plot(thresholds, recalls[:min_length], label='Recall')
            
              plt.xlabel('Threshold Value')
              plt.ylabel('Precision and Recall Value')
              plt.legend()
              plt.grid()
              st.pyplot(fig4)
              plt.close(fig4)
            except Exception as e:
              st.error(f"Error in precision-recall plotting: {str(e)}")
                
        else:  # Confusion Matrix
            with st.expander("🔢 Confusion Matrix", expanded=True):
                st.markdown("#### Model Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred_class)
                fig = plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                st.pyplot(fig)

        # Add explanation of metrics
        with st.expander("📚 Understanding the Metrics"):
             st.markdown("""
            #### Detailed Explanation of Evaluation Metrics
            
            1. *ROC-AUC Curve*
            - Plots true positive rate vs false positive rate
            - Higher AUC indicates better model discrimination
            - Perfect classifier would have AUC = 1.0
            
            2. *Model Loss Curve*
            - Shows how well the model is learning over time
            - Decreasing loss indicates improving model performance
            - Gap between training and validation loss helps identify overfitting
            
            3. *Model Accuracy Curve*
            - Tracks prediction accuracy during training
            - Higher accuracy indicates better model performance
            - Helps monitor potential overfitting
            
            4. *Precision-Recall Plot*
            - Shows trade-off between precision and recall
            - Helps in choosing optimal threshold for classification
            - Important for imbalanced datasets
            
            5. *Confusion Matrix*
            - Shows true positives, false positives, true negatives, and false negatives
            - Helps understand model's classification performance
            - Useful for identifying specific types of errors
            """)
