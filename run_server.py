import dill
import numpy as np
import pandas as pd
import flask

app = flask.Flask(__name__)
model = None


def load_model(model_path):
	global model
	with open(model_path, 'rb') as f:
		model = dill.load(f)


@app.route("/", methods=["GET"])
def general():
	return "Welcome to fraudelent prediction process"


@app.route("/predict", methods=["POST"])
def predict():
	data = {"success": False}

	if flask.request.method == "POST":
		dependents = np.nan
		property_area = np.nan
		applicant_income = np.nan
		coapplicant_income = np.nan
		loan_amount = np.nan
		loan_amount_term = np.nan
		gender = np.nan
		married = np.nan
		education = np.nan
		self_employed = np.nan
		credit_history = np.nan

		request_json = flask.request.get_json()
		if request_json['Dependents'] != '':
			dependents = request_json['Dependents']
		if request_json['Property_Area'] != '':
			property_area = request_json['Property_Area']
		if request_json['ApplicantIncome'] != '':
			applicant_income = request_json['ApplicantIncome']
		if request_json['CoapplicantIncome'] != '':
			coapplicant_income = request_json['CoapplicantIncome']
		if request_json['LoanAmount'] != '':
			loan_amount = request_json['LoanAmount']
		if request_json['Loan_Amount_Term'] != '':
			loan_amount_term = request_json['Loan_Amount_Term']
		if request_json['Gender'] != '':
			gender = request_json['Gender']
		if request_json['Married'] != '':
			married = request_json['Married']
		if request_json['Education'] != '':
			education = request_json['Education']
		if request_json['Self_Employed'] != '':
			self_employed = request_json['Self_Employed']
		if request_json['Credit_History'] != '':
			credit_history = request_json['Credit_History']

		preds = model.predict_proba(pd.DataFrame({'Gender': [gender],
												  'Married': [married],
												  'Dependents': [dependents],
												  'Education': [education],
												  'Self_Employed': [self_employed],
												  'ApplicantIncome': [applicant_income],
												  'CoapplicantIncome': [coapplicant_income],
												  'LoanAmount': [loan_amount],
												  'Loan_Amount_Term': [loan_amount_term],
												  'Credit_History': [credit_history],
												  'Property_Area': [property_area]
												  }))

		data["predictions"] = preds[:, 1][0]
		# indicate that the request was a success
		data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)


if __name__ == "__main__":
	print(("* Loading the model and Flask starting server..."
		"please wait until server has fully started"))
	modelpath = "models\catboost_class_pipeline.dill"
	load_model(modelpath)
	app.run()
