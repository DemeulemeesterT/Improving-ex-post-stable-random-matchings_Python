import pandas as pd
from src.Data import *

def read_data(file_path):
    df = pd.read_csv(file_path, delimiter="\t")  # Adjust delimiter if necessary

    # Extract unique students and schools
    ID_stud = sorted(df["child_id"].unique().tolist())
    ID_school = sorted(df["garten_id"].unique().tolist())

    # Create student preferences: Sort by `pref_nr` for each student
    pref = [[] for _ in ID_stud]
    for student in ID_stud:
        student_prefs = df[df["child_id"] == student].sort_values(by="pref_nr")["garten_id"].tolist()
        pref[ID_stud.index(student)] = student_prefs

    # Assign priorities: Schools rank students by their preference order
    prior = [[] for _ in ID_school]
    for school in ID_school:
        school_applicants = df[df["garten_id"] == school].sort_values(by="distance_m")["child_id"].tolist()
        prior[ID_school.index(school)] = [student for student in school_applicants]

    # Assign default capacities (modify as needed)
    cap = [10] * len(ID_school)  # Example: each school has a capacity of 10

    print(pref)
    print(prior[0])

    
    return Data(n_stud=len(ID_stud), 
                n_schools=len(ID_school), 
                pref=pref, 
                prior=prior, 
                cap=cap, 
                ID_stud=ID_stud, 
                ID_school=ID_school, 
                file_name=file_path)
