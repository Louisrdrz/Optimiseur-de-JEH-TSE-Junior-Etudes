import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import gurobipy as gp
from gurobipy import GRB

# Fonction modifiée pour utiliser Gurobi pour l'optimisation
def optimize_gurobi(budget_tot, nb_phase, intervenants, taux_phase):
    c = 42.34  # Coût de cotisations par JEH, ne change pas
    val = budget_tot  # Valeur objective ajustée au budget total

    model = gp.Model("nonlinear_mip")
    p = [model.addVar(lb=80, ub=450, name=f"p{i}") for i in range(nb_phase)]
    x = [model.addVar(vtype=GRB.INTEGER, lb=1, name=f"x{i}") for i in range(nb_phase)]
    y = [model.addVar(vtype=GRB.INTEGER, name=f"y{i}") for i in range(nb_phase)]

    for i in range(nb_phase):
        model.addConstr(p[i] * x[i] <= val * taux_phase[i], f"constraint_{i+1}")
        model.addConstr(x[i] == intervenants[i] * y[i], f"multiple_x{i+1}")

    model.addConstr(gp.quicksum(p[i] * x[i] for i in range(nb_phase)) <= val, "total_constraint")
    model.setObjective(gp.quicksum(p[i] * x[i] for i in range(nb_phase)) - c * gp.quicksum(x[i] for i in range(nb_phase)), GRB.MAXIMIZE)
    model.optimize()

    results = []
    if model.status == GRB.OPTIMAL:
        for i in range(nb_phase):
            results.append((x[i].X, round(p[i].X, 2)))  # (Nombre de JEH, Valeur par JEH)
    else:
        return None
    return results

def display_optimal_distribution(result):
    phase_names = []
    phase_values = []
    jeh_counts = []
    valeur = []
    count = 0

    for i, (n_jeh, valeur_jeh) in enumerate(result):
        st.markdown(f"**Phase {i + 1}**")
        st.info(f"Nombre de JEH : {n_jeh}")
        st.success(f"Valeur par JEH : {valeur_jeh:.2f} euros")  # Formatting for decimal values
        phase_names.append(f"Phase {i + 1}")
        phase_values.append(valeur_jeh)
        jeh_counts.append(n_jeh)
        valeur.append(n_jeh*valeur_jeh)

    count =sum(valeur)

    st.markdown(f"**Coût total de la mission HT : {count} euros**")


     # Plot the graph
    
    fig = go.Figure(data=[go.Bar(x=phase_names, y=phase_values)])
    fig.update_layout(title="Prix par phase en euros", xaxis_title="Phases", yaxis_title="Prix en euros")

    # Ajouter les annotations pour le nombre de JEH
    for i, value in enumerate(phase_values):
        fig.add_annotation(x=phase_names[i], y=value,
                            text=f"JEH: {jeh_counts[i]}",
                            showarrow=False,
                            yshift=10)

    st.plotly_chart(fig, use_container_width=True)

def display_invoice_table(result, ratio_frais_structure=0.05):
    # Create lists to hold the data
    designations = []
    nombre_de_jeh = []
    prix_unitaire_ht = []
    total_ht = []

    # Fill in the data based on the result
    for i, (n_jeh, valeur_jeh) in enumerate(result):
        designations.append(f"Phase {i + 1}")
        nombre_de_jeh.append(n_jeh)  # Assuming n_jeh is already an integer count
        prix_unitaire_ht.append(valeur_jeh)
        total_ht.append(round(n_jeh * valeur_jeh, 2))

    # Calculate total JEH and total amount
    total_jeh = sum(nombre_de_jeh)
    total_amount_ht = round(sum(total_ht), 2)
    

    # Add the totals
    designations.append("Nombre total de JEH")
    nombre_de_jeh.append(total_jeh)
    prix_unitaire_ht.append("")
    total_ht.append("")

    # Add the totals
    designations.append("")
    nombre_de_jeh.append("")
    prix_unitaire_ht.append("TOTAL HT hors frais")
    total_ht.append(total_amount_ht)

    
    # Calculate "Frais de structure"
    frais_structure = round(total_amount_ht * ratio_frais_structure, 2)


    # Insert "Frais de structure" before "TOTAL HT"
    designations.append("")
    nombre_de_jeh.append("")
    prix_unitaire_ht.append("Frais de stucture")
    total_ht.append(frais_structure)

    
    Total_ht =  total_amount_ht + frais_structure
    # Insert "TOTAL HT"
    designations.append("")
    nombre_de_jeh.append("")
    prix_unitaire_ht.append("Total HT")
    total_ht.append(Total_ht)


    # Add TVA and TOTAL TTC calculations
    tva = round(Total_ht * 0.2, 2)
    total_ttc = round(Total_ht + tva, 2)


    # Add TVA and TOTAL TTC calculations
    designations.append("")
    nombre_de_jeh.append("")
    prix_unitaire_ht.append("TVA 20% (à titre indicatif). Sur les encaissements")
    total_ht.append(tva)

    # Add TVA and TOTAL TTC calculations
    designations.append("")
    nombre_de_jeh.append("")
    prix_unitaire_ht.append("Total TTC")
    total_ht.append(total_ttc)

    # Create a DataFrame
    invoice_df = pd.DataFrame({
        'Désignation': designations,
        'Nombre de JEH': nombre_de_jeh,
        'Prix unitaire (HT)': prix_unitaire_ht,
        'TOTAL': total_ht
    })

    # Set the DataFrame's index to be the Désignation column
    invoice_df.set_index('Désignation', inplace=True)

    # Display the DataFrame as a table in Streamlit
    st.table(invoice_df)

    

# Main Streamlit app function
def main():
    st.title("Dashboard - Optimiseur de JEH (avec Gurobi)- TSE Junior Etudes")
    st.markdown("""Ce dashboard interactif est conçu pour faciliter l'analyse et la planification financière des missions de la TSE Junior Etudes en termes de JEH.""")

    multi = ''' Objectifs : 
    - maximiser le montant des JEH
    - minimiser le nombre de JEH
    - contrainte du budget global 
    - contrainte du budget par phase
    - les prix des JEH peuvent être différents par phase
    '''
    st.markdown(multi)

    col1, col2, col3 = st.columns([2, 2, 2])
    with col2:
        st.image("Logo TSE Junior Etudes.jpg", use_column_width=True)

    budget_tot = st.number_input("Entrer le budget total HT de la mission (multiple de 100)", value=5000, min_value=100, step=100)
    nb_phase = st.number_input("Entrer le nombre de phases", min_value=1, step=1, value=4)
    
    if nb_phase > 1:
        taux_phase = []
        lower_bound = 0.0
        for i in range(nb_phase - 1):
            upper_bound = st.slider(f"Fraction du budget jusqu'à la phase {i + 1}", 
                                    min_value=lower_bound, max_value=1.0, value=(lower_bound + 1.0/nb_phase), step=0.01)
            taux_phase.append(upper_bound - lower_bound)
            lower_bound = upper_bound
        taux_phase.append(1.0 - lower_bound)  

    elif nb_phase == 1:
        taux_phase = [1.0]  # Si il y a qu'une phase = tout le budget

    # Collecter le nombre d'intervenants par phase
    intervenants = []
    for i in range(nb_phase):
        intervenants.append(st.number_input(f"Nombre d'intervenants pour la phase {i+1}", min_value=1, step=1, value=2))

    if st.button("Optimiser"):
        result = optimize_gurobi(budget_tot, nb_phase, intervenants, taux_phase)
        if result:
            st.success("Optimisation réussie. Voici les résultats :")
            display_optimal_distribution(result)
            display_invoice_table(result)
            
            # Affichage supplémentaire des résultats, comme des graphiques, peut être ajouté ici
        else:
            st.error("Une erreur s'est produite lors de l'optimisation.")

if __name__ == "__main__":
    main()
