from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained decision tree model
with open('cattle_disease_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# List of symptoms and diseases
l1 = ['anorexia', 'abdominal_pain', 'anaemia', 'abortions', 'acetone', 'aggression', 'arthrogyposis',
      'ankylosis', 'anxiety', 'bellowing', 'blood_loss', 'blood_poisoning', 'blisters', 'colic', 'Condemnation_of_livers',
      'coughing', 'depression', 'discomfort', 'dyspnea', 'dysentery', 'diarrhoea', 'dehydration', 'drooling',
      'dull', 'decreased_fertility', 'diffculty_breath', 'emaciation', 'encephalitis', 'fever', 'facial_paralysis', 'frothing_of_mouth',
      'frothing', 'gaseous_stomach', 'highly_diarrhoea', 'high_pulse_rate', 'high_temp', 'high_proportion', 'hyperaemia', 'hydrocephalus',
      'isolation_from_herd', 'infertility', 'intermittent_fever', 'jaundice', 'ketosis', 'loss_of_appetite', 'lameness',
      'lack_of-coordination', 'lethargy', 'lacrimation', 'milk_flakes', 'milk_watery', 'milk_clots',
      'mild_diarrhoea', 'moaning', 'mucosal_lesions', 'milk_fever', 'nausea', 'nasel_discharges', 'oedema',
      'pain', 'painful_tongue', 'pneumonia', 'photo_sensitization', 'quivering_lips', 'reduction_milk_vields', 'rapid_breathing',
      'rumenstasis', 'reduced_rumination', 'reduced_fertility', 'reduced_fat', 'reduces_feed_intake', 'raised_breathing', 'stomach_pain',
      'salivation', 'stillbirths', 'shallow_breathing', 'swollen_pharyngeal', 'swelling', 'saliva', 'swollen_tongue',
      'tachycardia', 'torticollis', 'udder_swelling', 'udder_heat', 'udder_hardeness', 'udder_redness', 'udder_pain', 'unwillingness_to_move',
      'ulcers', 'vomiting', 'weight_loss', 'weakness']

disease = ['mastitis', 'blackleg', 'bloat', 'coccidiosis', 'cryptosporidiosis',
           'displaced_abomasum', 'gut_worms', 'listeriosis', 'liver_fluke', 'necrotic_enteritis', 'peri_weaning_diarrhoea',
           'rift_valley_fever', 'rumen_acidosis', 'traumatic_reticulitis', 'calf_diphtheria', 'foot_rot',
           'foot_and_mouth', 'ragwort_poisoning', 'wooden_tongue', 'infectious_bovine_rhinotracheitis',
           'acetonaemia', 'fatty_liver_syndrome', 'calf_pneumonia', 'schmallen_berg_virus', 'trypanosomosis', 'fog_fever']


@app.route('/')
def index():
    return render_template('index.html', symptoms=l1)


@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    cattle_id = request.form['cattle_id']
    symptoms = [request.form['symptom1'], request.form['symptom2'], request.form['symptom3'],
                request.form['symptom4'], request.form['symptom5']]

    # Prepare the input data for prediction
    input_symptoms = [0] * len(l1)
    for symptom in symptoms:
        if symptom != 'Select Here' and symptom in l1:
            input_symptoms[l1.index(symptom)] = 1

    # Reshape the input data for prediction
    input_data = [input_symptoms]

    # Make the prediction
    prediction = model.predict(input_data)
    predicted_disease = disease[prediction[0]]

    return render_template('result.html', cattle_id=cattle_id, disease=predicted_disease)


if __name__ == '__main__':
    app.run(debug=True)
