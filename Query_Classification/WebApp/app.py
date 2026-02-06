from flask import Flask, request, render_template, redirect, url_for
from modelrouter import ModelHandler  
from download_models import download_and_extract
import os

if not os.path.exists("WebApp/models"):
    download_and_extract(
        file_id="1LR9Vr4dkl8C6uYt8KfnenK0qgXD-dVS5",
        output_zip="models.zip",
        extract_to="WebApp/models"
    )


app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route('/')
def home():
    return redirect(url_for('form'))

@app.route('/form', methods=['GET', 'POST'])
def form():
    result = {}
    if request.method == 'POST':
        query = request.form['query']
        lang = request.form['lang']
        print(lang)  # 'eng', 'hin', 'guj', or 'multi'

        try:
            handler = ModelHandler(lang)
            predicted_label = handler.predict(query)
            print(f"Predicted label: {predicted_label}")

            result = {
                "query": query,
                "language": lang,
                "label": predicted_label  # Direct label from model
            }
        except Exception as e:
            result = {"error": str(e)}

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
