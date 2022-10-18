import sklearn
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

symptom_entries = []
array_of_symptoms = np.array([["itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering", "chills", "joint_pain", "stomach_pain", "acidity",
"ulcers_on_tongue", "muscle_wasting", "vomiting", "burning_micturition", "spotting_urination", "fatigue", "weight_gain", "anxiety",
"cold_hands_and_feet", "mood_swings", "weight_loss", "restlessness", "lethargy", "patches_in_throat", "irregular_sugar_level",
"cough", "high_fever", "sunken_eyes", "breathlessness", "sweating", "dehydration", "indigestion", "headache", "yellowish_skin", "dark_urine",
"nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "constipation", "abdominal_pain", "diarrhoea", "mild_fever",
"yellow_urine", "yellowing_of_eyes", "acute_liver_failure", "fluid_overload", "swelling_of_stomach", "swelled_lymph_nodes", "malaise",
"blurred_and_distorted_vision", "phlegm", "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion", "chest_pain",
"weakness_in_limbs", "fast_heart_rate", "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", "irritation_in_anus",
"neck_pain", "dizziness", "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels", "puffy_face_and_eyes", "enlarged_thyroid",
"brittle_nails", "swollen_extremeties", "excessive_hunger", "extra_marital_contacts", "drying_and_tingling_lips", "slurred_speech", "knee_pain",
"hip_joint_pain", "muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness", "spinning_movements", "loss_of_balance",
"unsteadiness", "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort", "foul_smell_of_urine", "continuous_feel_of_urine",
"passage_of_gases", "internal_itching", "toxic_look_typhos", "depression", "irritability", "muscle_pain", "altered_sensorium",
"red_spots_over_body", "belly_pain", "abnormal_menstruation", "dischromic_patches", "watering_from_eyes", "increased_appetite", "polyuria", "family_history",
"mucoid_sputum", "rusty_sputum", "lack_of_concentration", "visual_disturbances", "receiving_blood_transfusion", "receiving_unsterile_injections",
"coma", "stomach_bleeding", "distention_of_abdomen", "history_of_alcohol_consumption", "blood_in_sputum", "prominent_veins_on_calf",
"palpitations", "painful_walking", "pus_filled_pimples", "blackheads", "scurring", "skin_peeling", "silver_like_dusting", "small_dents_in_nails",
"inflammatory_nails", "blister", "red_sore_around_nose", "yellow_crust_ooze"]])
done = False
while done != True:
    symptom_entry = input("Please enter a symptom (If you have no other symptoms enter 'stop'): ").lower().replace(" ", "_")
    if symptom_entry in array_of_symptoms:
        symptom_entries.append(symptom_entry)
    elif symptom_entry == "stop":
        done = True
    else:
        print("Symptom is not in our database.")

for i in range(len(array_of_symptoms)):
    for j in range(len(array_of_symptoms[i])):
        if array_of_symptoms[i][j] in symptom_entries:
            array_of_symptoms[i][j] = 1
        else:
            array_of_symptoms[i][j] = 0

user_symptoms = [list(map(int, i)) for i in array_of_symptoms]

# import our train and test data
data = pd.read_csv("data.csv")

# Create a list of each column in the dataframe
le = preprocessing.LabelEncoder()
itching = list(data["itching"])
skin_rash = list(data["skin_rash"])
nodal_skin_eruptions = list(data["nodal_skin_eruptions"])
continuous_sneezing = list(data["continuous_sneezing"])
shivering = list(data["shivering"])
chills = list(data["chills"])
joint_pain = list(data["joint_pain"])
stomach_pain = list(data["stomach_pain"])
acidity = list(data["acidity"])
ulcers_on_tongue = list(data["ulcers_on_tongue"])
muscle_wasting = list(data["muscle_wasting"])
vomiting = list(data["vomiting"])
burning_micturition = list(data["burning_micturition"])
spotting_urination = list(data["spotting_urination"])
fatigue = list(data["fatigue"])
weight_gain = list(data["weight_gain"])
anxiety = list(data["anxiety"])
cold_hands_and_feet = list(data["cold_hands_and_feet"])
mood_swings = list(data["mood_swings"])
weight_loss = list(data["weight_loss"])
restlessness = list(data["restlessness"])
lethargy = list(data["lethargy"])
patches_in_throat = list(data["patches_in_throat"])
irregular_sugar_level = list(data["irregular_sugar_level"])
cough = list(data["cough"])
high_fever = list(data["high_fever"])
sunken_eyes = list(data["sunken_eyes"])
breathlessness = list(data["breathlessness"])
sweating = list(data["sweating"])
dehydration = list(data["dehydration"])
indigestion = list(data["indigestion"])
headache = list(data["headache"])
yellowish_skin = list(data["yellowish_skin"])
dark_urine = list(data["dark_urine"])
nausea = list(data["nausea"])
loss_of_appetite = list(data["loss_of_appetite"])
pain_behind_the_eyes = list(data["pain_behind_the_eyes"])
back_pain = list(data["back_pain"])
constipation = list(data["constipation"])
abdominal_pain = list(data["abdominal_pain"])
diarrhoea = list(data["diarrhoea"])
mild_fever = list(data["mild_fever"])
yellow_urine = list(data["yellow_urine"])
yellowing_of_eyes = list(data["yellowing_of_eyes"])
acute_liver_failure = list(data["acute_liver_failure"])
fluid_overload = list(data["fluid_overload"])
swelling_of_stomach = list(data["swelling_of_stomach"])
swelled_lymph_nodes = list(data["swelled_lymph_nodes"])
malaise = list(data["malaise"])
blurred_and_distorted_vision = list(data["blurred_and_distorted_vision"])
phlegm = list(data["phlegm"])
throat_irritation = list(data["throat_irritation"])
redness_of_eyes = list(data["redness_of_eyes"])
sinus_pressure = list(data["sinus_pressure"])
runny_nose = list(data["runny_nose"])
congestion = list(data["congestion"])
chest_pain = list(data["chest_pain"])
weakness_in_limbs = list(data["weakness_in_limbs"])
fast_heart_rate = list(data["fast_heart_rate"])
pain_during_bowel_movements = list(data["pain_during_bowel_movements"])
pain_in_anal_region = list(data["pain_in_anal_region"])
bloody_stool = list(data["bloody_stool"])
irritation_in_anus = list(data["irritation_in_anus"])
neck_pain = list(data["neck_pain"])
dizziness = list(data["dizziness"])
cramps = list(data["cramps"])
bruising = list(data["bruising"])
obesity = list(data["obesity"])
swollen_legs = list(data["swollen_legs"])
swollen_blood_vessels = list(data["swollen_blood_vessels"])
puffy_face_and_eyes = list(data["puffy_face_and_eyes"])
enlarged_thyroid = list(data["enlarged_thyroid"])
brittle_nails = list(data["brittle_nails"])
swollen_extremeties = list(data["swollen_extremeties"])
excessive_hunger = list(data["excessive_hunger"])
extra_marital_contacts = list(data["extra_marital_contacts"])
drying_and_tingling_lips = list(data["drying_and_tingling_lips"])
slurred_speech = list(data["slurred_speech"])
knee_pain = list(data["knee_pain"])
hip_joint_pain = list(data["hip_joint_pain"])
muscle_weakness = list(data["muscle_weakness"])
stiff_neck = list(data["stiff_neck"])
swelling_joints = list(data["swelling_joints"])
movement_stiffness = list(data["movement_stiffness"])
spinning_movements = list(data["spinning_movements"])
loss_of_balance = list(data["loss_of_balance"])
unsteadiness = list(data["unsteadiness"])
weakness_of_one_body_side = list(data["weakness_of_one_body_side"])
loss_of_smell = list(data["loss_of_smell"])
bladder_discomfort = list(data["bladder_discomfort"])
foul_smell_of_urine = list(data["foul_smell_of_urine"])
continuous_feel_of_urine = list(data["continuous_feel_of_urine"])
passage_of_gases = list(data["passage_of_gases"])
internal_itching = list(data["internal_itching"])
toxic_look_typhos = list(data["toxic_look_typhos"])
depression = list(data["depression"])
irritability = list(data["irritability"])
muscle_pain = list(data["muscle_pain"])
altered_sensorium = list(data["altered_sensorium"])
red_spots_over_body = list(data["red_spots_over_body"])
belly_pain = list(data["belly_pain"])
abnormal_menstruation = list(data["abnormal_menstruation"])
dischromic_patches = list(data["dischromic_patches"])
watering_from_eyes = list(data["watering_from_eyes"])
increased_appetite = list(data["increased_appetite"])
polyuria = list(data["polyuria"])
family_history = list(data["family_history"])
mucoid_sputum = list(data["mucoid_sputum"])
rusty_sputum = list(data["rusty_sputum"])
lack_of_concentration = list(data["lack_of_concentration"])
visual_disturbances = list(data["visual_disturbances"])
receiving_blood_transfusion = list(data["receiving_blood_transfusion"])
receiving_unsterile_injections = list(data["receiving_unsterile_injections"])
coma = list(data["coma"])
stomach_bleeding = list(data["stomach_bleeding"])
distention_of_abdomen = list(data["distention_of_abdomen"])
history_of_alcohol_consumption = list(data["history_of_alcohol_consumption"])
blood_in_sputum = list(data["blood_in_sputum"])
prominent_veins_on_calf = list(data["prominent_veins_on_calf"])
palpitations = list(data["palpitations"])
painful_walking = list(data["painful_walking"])
pus_filled_pimples = list(data["pus_filled_pimples"])
blackheads = list(data["blackheads"])
scurring = list(data["scurring"])
skin_peeling = list(data["skin_peeling"])
silver_like_dusting = list(data["silver_like_dusting"])
small_dents_in_nails = list(data["small_dents_in_nails"])
inflammatory_nails = list(data["inflammatory_nails"])
blister = list(data["blister"])
red_sore_around_nose = list(data["red_sore_around_nose"])
yellow_crust_ooze = list(data["yellow_crust_ooze"])
prognosis = le.fit_transform(list(data["prognosis"]))

X = list(zip(itching, skin_rash, nodal_skin_eruptions, continuous_sneezing, shivering, chills, joint_pain, stomach_pain, acidity,
             ulcers_on_tongue, muscle_wasting, vomiting, burning_micturition, spotting_urination, fatigue, weight_gain, anxiety,
             cold_hands_and_feet, mood_swings, weight_loss, restlessness, lethargy, patches_in_throat, irregular_sugar_level,
             cough, high_fever, sunken_eyes, breathlessness, sweating, dehydration, indigestion, headache, yellowish_skin, dark_urine,
             nausea, loss_of_appetite, pain_behind_the_eyes, back_pain, constipation, abdominal_pain, diarrhoea, mild_fever,
             yellow_urine, yellowing_of_eyes, acute_liver_failure, fluid_overload, swelling_of_stomach, swelled_lymph_nodes, malaise,
             blurred_and_distorted_vision, phlegm, throat_irritation, redness_of_eyes, sinus_pressure, runny_nose, congestion, chest_pain,
             weakness_in_limbs, fast_heart_rate, pain_during_bowel_movements, pain_in_anal_region, bloody_stool, irritation_in_anus,
             neck_pain, dizziness, cramps, bruising, obesity, swollen_legs, swollen_blood_vessels, puffy_face_and_eyes, enlarged_thyroid,
             brittle_nails, swollen_extremeties, excessive_hunger, extra_marital_contacts, drying_and_tingling_lips, slurred_speech, knee_pain,
             hip_joint_pain, muscle_weakness, stiff_neck, swelling_joints, movement_stiffness, spinning_movements, loss_of_balance,
             unsteadiness, weakness_of_one_body_side, loss_of_smell, bladder_discomfort, foul_smell_of_urine, continuous_feel_of_urine,
             passage_of_gases, internal_itching, toxic_look_typhos, depression, irritability, muscle_pain, altered_sensorium,
             red_spots_over_body, belly_pain, abnormal_menstruation, dischromic_patches, watering_from_eyes, increased_appetite, polyuria, family_history,
             mucoid_sputum, rusty_sputum, lack_of_concentration, visual_disturbances, receiving_blood_transfusion, receiving_unsterile_injections,
             coma, stomach_bleeding, distention_of_abdomen, history_of_alcohol_consumption, blood_in_sputum, prominent_veins_on_calf,
             palpitations, painful_walking, pus_filled_pimples, blackheads, scurring, skin_peeling, silver_like_dusting, small_dents_in_nails,
             inflammatory_nails, blister, red_sore_around_nose, yellow_crust_ooze))
y = list(prognosis)

# What are we predicting
predict = "prognosis"

# create our training and testing variables
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.05)

# Create our model object
model = KNeighborsClassifier(n_neighbors=5)

# Train our model
model.fit(X_train, y_train)

# Check how accurate our model is and print it
# acc = model.score(X_test, y_test)
# print(acc)

# Predict using our model
predicted = model.predict(user_symptoms)

# Create a list of all of our different prognoses
# Then print out for every test of the model what the prediction was
prognosis_list = [*set(list(data["prognosis"]))]
for x in range(len(predicted)):
    print("The AI Doctor thinks you may have: ", prognosis_list[predicted[x]])
print("Consult a doctor before treatment.")
