# 📩 Spam Email Prediction (Streamlit + Machine Learning)

A Streamlit web app that predicts whether an email is SPAM or HAM (Not Spam) using a trained ML model saved as `spam_model.joblib`.

Project files: `app.py` (Streamlit UI + inference), `train.py` (training), `requirements.txt` (deps), `mail_data.csv` (dataset/sample), `Spam_Mail_Prediction_using_Machine_Learning.ipynb` (notebook), `spam_model.joblib` (model), `.gitignore` (ignores venv).

🚀 Run the project (exact commands) — Windows setup (as requested):
```bash
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
