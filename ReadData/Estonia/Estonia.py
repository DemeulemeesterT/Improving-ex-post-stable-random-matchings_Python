import pandas as pd
from src.Data import *
import csv

# Coded with help of ChatGPT


def read_dataEstonia(file_path):
    df = pd.read_csv(file_path, delimiter="\t")  # Adjust delimiter if necessary

    # Extract unique students and schools
    ID_stud = sorted(df["child_id"].unique().tolist())
    ID_school = sorted(df["garten_id"].unique().tolist())

    # Create student preferences: Sort by `pref_nr` for each student
    pref = [[] for _ in ID_stud]
    for student in ID_stud:
        student_prefs = df[df["child_id"] == student].sort_values(by="pref_nr")["garten_id"].tolist()
        pref[ID_stud.index(student)] = student_prefs

    # Determine walking groups
    #tresholds = [1000, 2000, 4000, 7000]
    tresholds = [4000]
    n_groups = len(tresholds) + 1

    

    # Determine which group students belong to for each school
    prior = [[] for _ in ID_school]
    for school in ID_school:
        groups = []
        for _ in range(n_groups):
            groups.append([])
        for student in ID_stud:
            filtered_values = df.loc[(df["child_id"] == student) & (df["garten_id"] == school), "distance_m"]

            # Check if there is at least one result
            if not filtered_values.empty:
                dist = filtered_values.iloc[0]  # Now it's safe
                #print("Stud, school, dist", student, school, dist)
                if dist <= tresholds[0]:
                    groups[0].append(student)
                else:
                    for k in range(1, len(tresholds)):
                        if (tresholds[k-1] <= dist) & (dist <= tresholds[k]):
                            #print("\tTresholds", tresholds[k-1], tresholds[k])
                            groups[k].append(student)
                    if dist >= tresholds[-1]:
                        groups[n_groups-1].append(student)


        # Now create tuples from the groups that contain more than 1 student
        tuplegroups = [
            group[0] if len(group) == 1 else tuple(group) 
            for group in groups 
            if len(group) > 0  # This ensures empty groups are skipped
        ]

        #print("Tuplegroups", tuplegroups)
        # Store in prior
        for group in tuplegroups:
            prior[ID_school.index(school)].append([group] if isinstance(group, np.ndarray) else group)
    
        #tuplegroups = []
        #for _ in range(n_groups):
        #    tuplegroups.append([])
        #for k in range(n_groups):
        #    if len(groups[k]) == 1:
        #        tuplegroups[k] = groups[k]
        #        # Add groups to prior
        #        prior[ID_school.index(school)].append([tuplegroups[k]])
        #    elif len(groups[k]) >= 1:
        #        tuplegroups[k] = tuple(groups[k])
        #        # Add groups to prior
        #        prior[ID_school.index(school)].append(tuplegroups[k])
        #    # Don't do anything if size == 0

            



    # Assign priorities: Schools rank students by their preference order
    #prior = [[] for _ in ID_school]
    #for school in ID_school:
    #    school_applicants = df[df["garten_id"] == school].sort_values(by="distance_m")["child_id"].tolist()
    #    prior[ID_school.index(school)] = school_applicants

    # Capacities
    cap = [20, 20, 34, 18, 20, 38, 5]
    #print("Prior", prior)
    #print("Pref", pref)
    
    return Data(n_stud=len(ID_stud), 
                n_schools=len(ID_school), 
                pref=pref, 
                prior=prior, 
                cap=cap, 
                ID_stud=ID_stud, 
                ID_school=ID_school, 
                file_name=file_path)
