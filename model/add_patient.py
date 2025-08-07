from datetime import datetime


class Patient:
    def __init__(self, patient_id, name, age, date_of_birth, gender, education_years,
                 language, handedness, clinician_name):
        self.patient_id = patient_id # patient details
        self.name = name
        self.age = age
        self.date_of_birth = date_of_birth
        self.gender = gender
        self.education_years = education_years
        self.language = language
        self.handedness = handedness
        self.test_date = datetime.today().strftime('%Y-%m-%d')
        self.clinician_name = clinician_name
        self.cognitive_score = 0

    def display_info(self):
        print("Patient Details")
        print("------------------")
        print(f"Patient ID      : {self.patient_id}")
        print(f"Name            : {self.name}")
        print(f"Age             : {self.age}")
        print(f"Date of Birth   : {self.date_of_birth}")
        print(f"Gender          : {self.gender}")
        print(f"Education (yrs) : {self.education_years}")
        print(f"Language        : {self.language}")
        print(f"Handedness      : {self.handedness}")
        print(f"Test Date       : {self.test_date}")
        print(f"Clinician Name  : {self.clinician_name}")
        print(f"Cognitive Score : {self.cognitive_score}")
def get_patient_input():
    print("Enter Patient Details")
    print("----------------------")
    patient_id = input("Patient ID: ")
    name = input("Full Name: ")
    age = int(input("Age: "))
    date_of_birth = input("Date of Birth (YYYY-MM-DD): ")
    gender = input("Gender (Male/Female/Other): ")
    education_years = int(input("Years of Education: "))
    language = input("Primary Language: ")
    handedness = input("Handedness (Right/Left/Ambidextrous): ")
    clinician_name = input("Clinician Name: ")


    return Patient(patient_id, name, age, date_of_birth, gender, education_years,
                   language, handedness, clinician_name)