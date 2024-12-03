from flask import Flask, render_template, request, redirect, url_for
from Model.Churn_Prediction import predict_churn

app = Flask(__name__ )

# @app.route("/", methods=['GET'])
# def main():
#     return render_template("main.html")

@app.route('/', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == "POST":
        try :
            user_input = {
            "gender": request.form.get("gender"),
            "seniorCitizen": int(request.form.get("seniorCitizen")),
            "Partner": request.form.get("partner"),
            "Dependents": request.form.get("dependents"),
            "tenure": float(request.form.get("tenure")),
            "PhoneService": request.form.get("phoneService"),
            "MultipleLines": request.form.get("multipleLines"),
            "InternetService": request.form.get("internetService"),
            "OnlineSecurity": request.form.get("onlineSecurity"),
            "OnlineBackup": request.form.get("onlineBackup"),
            "DeviceProtection": request.form.get("deviceProtection"),
            "TechSupport": request.form.get("techSupport"),
            "StreamingTV": request.form.get("streamingTV"),
            "StreamingMovies": request.form.get("streamingMovies"),
            "Contract": request.form.get("contract"),
            "PaperlessBilling": request.form.get("paperlessBilling"),
            "PaymentMethod": request.form.get("paymentMethod"),
            "MonthlyCharges": float(request.form.get("monthlyCharges")),
            "TotalCharges": float(request.form.get("totalCharges"))
            }
            result = predict_churn(user_input)
            return redirect(url_for('result', prediction=result))
        except Exception as e:
            return f"Error: {str(e)}"
        
    return render_template("main.html")

@app.route('/result')
def result():
    # Ambil hasil prediksi dari query parameter
    prediction = request.args.get('prediction')
    return render_template("result.html", prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True) 