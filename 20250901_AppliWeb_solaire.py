# -*- coding: utf-8 -*-
"""
Créé le Mer 17 janv. 18:17:53 2024

@auteur: Gemini
"""
# =========================================================================== #
#       IMPORTATION DES LIBRAIRIES PYTHON POUR FAIRE TOURNER LE PROGRAMME     #
# =========================================================================== #
import streamlit as st
import numpy as np
import pandas as pd
from numpy import linalg as LA
from scipy import interpolate
import plotly.express as px
import plotly.graph_objects as go
import datetime

# =========================================================================== #
#                 CONVERSIONS                                                 #
# =========================================================================== #
DEG2RAD = np.pi/180.0
HOUR2RAD = 2 * np.pi / 24
DAY2RAD = 2 * np.pi / 365
HOUR2SEC = 3600

# =========================================================================== #
#           INTERFACE UTILISATEUR AVEC STREAMLIT                              #
# =========================================================================== #
st.set_page_config(layout="wide")
st.title("Simulation de production d'énergie solaire ☀️")

st.markdown("""
Cette application permet de simuler la production d'énergie d'une installation solaire thermique ou photovoltaïque. 
Vous pouvez ajuster les paramètres ci-dessous pour voir comment ils affectent les résultats.
""")

st.markdown("[📄 Documentation du code (PDF)](https://github.com/RemiCoisnon/solaire/raw/main/documentation.pdf)", unsafe_allow_html=True)

# Colonnes pour organiser l'interface
col1, col2 = st.columns(2)

with col1:
    st.header("Paramètres de calculs")
    st.subheader("Plus la valeur est élevée, plus le code est précis... et lent!")
    # New Slider for nb_h
    nb_h = st.slider("Nombre de points de calcul par jour", min_value=5, max_value=1000, value=300, step=50)

    st.header("Paramètres de l'installation")
    system_type = st.radio(
        "Type de système",
        ("Photovoltaïque", "Solaire thermique"),
        key="system_type"
    )
    analyse_annee = st.checkbox("Effectuer une analyse sur une année complète", value=False)
    
    if analyse_annee:
        d_vec_input = np.arange(0, 365, 1)
    else:
        # Utiliser une date par défaut cohérente
        default_date = datetime.date(2025, 10, 21)
        selected_date = st.date_input("Sélectionnez le jour à analyser", value=default_date)
        d_vec_input = [selected_date.timetuple().tm_yday + 10]

    my_lambda_deg = st.slider("Latitude (°)", -90, 90, -43)
    my_lambda = my_lambda_deg * DEG2RAD
    
    # Initialisation de l'état de session pour les panneaux
    if 'panels' not in st.session_state:
        st.session_state.panels = []
    
    # Gérer la liste des panneaux en fonction du type de système
    if 'last_system_type' not in st.session_state or st.session_state.last_system_type != system_type:
        st.session_state.panels = []
        if system_type == "Photovoltaïque":
            st.session_state.panels.append({"beta": -90, "theta": 20, "eta": 25, "surface": 6})
            st.session_state.panels.append({"beta": 90, "theta": 20, "eta": 25, "surface": 6})
        else:
            st.session_state.panels.append({"beta": 0, "theta": 90, "eta": 78, "surface": 4.6})
        st.session_state.last_system_type = system_type

    st.subheader(f"Paramètres des panneaux {system_type.lower()}")
    
    def add_panel():
        if system_type == "Photovoltaïque":
            st.session_state.panels.append({"beta": 0, "theta": 20, "eta": 25, "surface": 6})
        else:
            st.session_state.panels.append({"beta": 0, "theta": 90, "eta": 78, "surface": 4.6})

    def remove_panel(index):
        st.session_state.panels.pop(index)

    # Afficher les panneaux et leurs paramètres de manière dynamique
    betap_vec = []
    thetap_vec = []
    eta_vec = []
    surface_vec = []

    for i, panel in enumerate(st.session_state.panels):
        with st.container():
            st.markdown(f"""
            <div style="border: 1px solid #ccc; padding: 15px; border-radius: 10px; background-color: #f9f9f9;">
                <h4>🪟 Surface {i+1}</h4>
            """,
            unsafe_allow_html=True)
        
            # Ligne 1 : sliders étendus
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**Orientation**  \n_↔ Est (-90°) — Sud (0°) — Ouest (+90°)_")
                panel["beta"] = st.slider(" ", min_value=-90, max_value=90, value=int(panel["beta"]), step=1, key=f"beta_{i}")
        
            with col2:
                st.markdown("**Inclinaison**  \n_0° = Horizontal — 90° = Vertical_")
                panel["theta"] = st.slider(" ", min_value=0, max_value=90, value=int(panel["theta"]), step=1, key=f"theta_{i}")
        
            # Ligne 2 : rendement + surface
            col3, col4 = st.columns([1, 1])
            with col3:
                panel["eta"] = st.number_input("Rendement (%)", value=panel["eta"], key=f"eta_{i}")
            with col4:
                panel["surface"] = st.number_input("Surface (m²)", value=panel["surface"], key=f"surface_{i}")
        
            # Bouton pour supprimer
            st.button("❌ Supprimer ce panneau", on_click=remove_panel, args=(i,), key=f"remove_{i}")
        
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("---")
        
        betap_vec.append(panel["beta"])
        thetap_vec.append(panel["theta"])
        eta_vec.append(panel["eta"])
        surface_vec.append(panel["surface"])

    st.button("Ajouter une surface de panneaux", on_click=add_panel)


with col2:
    if system_type == "Solaire thermique":
        st.header("Paramètres du système de chauffage")
        # Ballons
        Cp = st.number_input("Capacité calorifique de l'eau (J/(K.kg))", value=4180.0)
        MTAMPON = st.number_input("Masse d'eau dans le ballon tampon (kg)", value=200.0)
        MCUMULUS = st.number_input("Masse d'eau dans le ballon cumulus (kg)", value=150.0)
        SBALLON = st.number_input("Surface du ballon, pour mesure pertes caloriques (m²)", value=1.7)
        EBALLON = st.number_input("Épaisseur de l'isolant du ballon (m)", value=6e-2, format="%e")
        #
        TCAVE = st.slider("Température intérieure du local du ballon (°C)", -5, 30, 10)
        TINT = st.slider("Température intérieure de la maison (°C)", -5, 30, 19)
        TEXTCHAUFFE = st.slider("Température extérieure déclenchant le chauffage (°C)", 0, 30, 17)
        #
        P0 = st.number_input("Puissance nominale radiateur (W)", value=2 * 1782.0)
        DELTAT0 = st.number_input("Radiateur: Écart de température nominal (°C)", value=50.0)
        GAMMA = st.number_input("Pente de la courbe radiateur", value=1.3)
        #
        P_POMPE_CIRCU = st.number_input("Puissance de la pompe du circulateur (W)", value=150.0)
        LTUYAUX = st.number_input("Longueur des tuyaux (m)", value=20.0)
        ETUYAUX = st.number_input("Épaisseur de l'isolant des tuyaux (m)", value=2e-2, format="%e")
        STUYAUX = LTUYAUX * np.pi * (2 * 1.6e-2)
        LAMBDA = st.number_input("Conductivité thermique de l'isolant (W/(m.K))", value=4.2e-2, format="%e")
        MECS = st.number_input("Masse d'eau chaude consommée par jour (kg)", value=100.0)
        TIMEECS_start = st.slider("Début d'utilisation de l'eau chaude (h)", 0, 24, 19, key="start")
        TIMEECS_end = st.slider("Fin d'utilisation de l'eau chaude (h)", 0, 24, 20, key="end")
        TIMEECS = [TIMEECS_start, TIMEECS_end]
        is_thermique = True
    else:
        # Initialisation pour éviter les erreurs de variables non définies
        Cp, MTAMPON, MCUMULUS, LAMBDA, SBALLON, EBALLON, TCAVE, TINT = 0, 0, 0, 0, 0, 0, 0, 0
        TEXTCHAUFFE, P0, DELTAT0, GAMMA, P_POMPE_CIRCU, LTUYAUX, ETUYAUX, STUYAUX = 0, 0, 0, 0, 0, 0, 0, 0
        MECS, TIMEECS, is_thermique = 0, [0, 0], False

# =========================================================================== #
#           PARAMÈTRES INITIAUX ET CONSTANTES                                 #
# =========================================================================== #
PAMONT = 1300  # Puissance solaire en amont de l'atmosphère [Watts/m2]
R = 6300.  # km Rayon de la Terre
H = 500.  # km Épaisseur de l'atmosphère
theta0 = 23.5 * DEG2RAD  # Inclinaison axe de rotation Terre
R2 = R + H

# =========================================================================== #
#                   FONCTIONS DE CALCULS                                    #
# =========================================================================== #

def rotation_x(psi):
    """Effectue une rotation autour de l'axe x."""
    cp = np.cos(psi)
    sp = np.sin(psi)
    L = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])
    return L

def rotation_y(psi):
    """Effectue une rotation autour de l'axe y."""
    cp = np.cos(psi)
    sp = np.sin(psi)
    L = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return L

def rotation_z(psi):
    """Effectue une rotation autour de l'axe z."""
    cp = np.cos(psi)
    sp = np.sin(psi)
    L = np.array([[cp, sp, 0], [-sp, cp, 0], [0, 0, 1]])
    return L

def my_rot(axis, angle):
    """Retourne la matrice de rotation pour un axe donné."""
    if axis == "x":
        L = rotation_x(angle)
    elif axis == "y":
        L = rotation_y(angle)
    elif axis == "z":
        L = rotation_z(angle)
    else:
        st.error("Axe de rotation non correct.")
        st.stop()
    return L

def rotation_st_3(d, theta0, h, my_lambda):
    """Calcul de la matrice de rotation pour le repère terrestre incliné."""
    Lstsf = my_rot("z", -d)
    Lsf1 = my_rot("y", theta0)
    L12 = my_rot("z", h)
    L23 = my_rot("y", my_lambda)
    Lst3 = (((L23).dot(L12)).dot(Lsf1)).dot(Lstsf)
    return Lst3

def rotation_st_2(d, theta0, h):
    """Calcul de la matrice de rotation pour le repère tournant à l'heure h."""
    Lstsf = my_rot("z", -d)
    Lsf1 = my_rot("y", theta0)
    L12 = my_rot("z", h)
    Lst2 = ((L12).dot(Lsf1)).dot(Lstsf)
    return Lst2

def rotation_2_5(my_lambda, betap, thetap):
    """Calcul de la matrice de rotation pour le repère du panneau."""
    L23 = my_rot("y", my_lambda)
    L34 = my_rot("x", betap)
    L45 = my_rot("y", thetap)
    L25 = (L45.dot(L34)).dot(L23)
    return L25

def fun_absorption(epaisseur):
    """Calcule l'absorption atmosphérique en fonction de l'épaisseur."""
    epaisseur_vec = [500, 541, 553, 590, 661, 780, 979, 1317, 1880, 2563, 2600, 10000]
    absorption_vec = [11, 23, 25, 30, 36, 44, 52, 63, 76, 98, 100, 100]
    f = interpolate.interp1d(epaisseur_vec, absorption_vec)
    absorption = f(epaisseur)
    return absorption

def fun_temperature_exterieur():
    """Détermine la température extérieure moyenne en fonction du jour de l'année."""
    days_vec = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    temperature_vec = [5.8, 6.3, 10, 13.5, 17.6, 22.5, 25, 24.5, 20, 15.8, 10.1, 6.6, 5.8]
    days1 = np.cumsum(days_vec)
    f_temperature_ext = interpolate.interp1d(days1, temperature_vec)
    return f_temperature_ext

def day_month(my_day):
    """Associe un jour et un mois à un numéro de jour annuel."""
    days_vec = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_vec = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']
    days1 = np.cumsum(days_vec)
    day_sun = my_day - 10
    if day_sun > 0:
        pass
    else:
        day_sun = 365 + day_sun
    tt = day_sun > days1
    rr = [ind for ind, item in enumerate(tt) if item]
    i_month = rr[-1]
    jour = day_sun - days1[i_month]
    mois = month_vec[i_month]
    return jour, mois


# =========================================================================== #
#           LOGIQUE DE CALCUL (RÉPONSE DU BACKEND)                            #
# =========================================================================== #
@st.cache_data
def run_simulation(betap_vec, thetap_vec, eta_vec, surface_vec, d_vec, my_lambda,
                   Cp, MTAMPON, MCUMULUS, LAMBDA, SBALLON, EBALLON, TCAVE, TINT, 
                   TEXTCHAUFFE, P0, DELTAT0, GAMMA, P_POMPE_CIRCU, LTUYAUX, ETUYAUX, STUYAUX, 
                   MECS, TIMEECS, analyse_annee, is_thermique, nb_h):
    
    # Recalculate hvec and dt here, inside the cached function, to ensure it's
    # part of the cache key and updates when nb_h changes.
    hvec = np.linspace(0, 24, nb_h)
    dt = hvec[1] - hvec[0]

    STst = [1, 0, 0]  # Vecteur Soleil vers Terre dans le repère soleil tournant
    
    # Initialisation des vecteurs de résultats
    anglevec_p = np.zeros((nb_h, len(d_vec), len(betap_vec)))
    p_disponible_per_panel = np.zeros((nb_h, len(d_vec), len(betap_vec)))
    p_aval_vec = np.zeros((nb_h, len(d_vec)))

    # Variables cumulatives pour l'ensemble du système
    p_disponible_total = np.zeros((nb_h, len(d_vec)))
    p_radiateur_total = np.zeros((nb_h, len(d_vec)))
    p_cumulus_total = np.zeros((nb_h, len(d_vec)))
    p_tampon_total = np.zeros((nb_h, len(d_vec)))
    p_utile_total = np.zeros((nb_h, len(d_vec)))
    p_circulateur_total = np.zeros((nb_h, len(d_vec)))
    p_ecs_total = np.zeros((nb_h, len(d_vec)))
    p_tuyaux_total = np.zeros((nb_h, len(d_vec)))
    t_ballon_tampon_total = np.zeros((nb_h, len(d_vec)))
    t_cumulus_total = np.zeros((nb_h, len(d_vec)))
    p_tampon_fuite_total = np.zeros((nb_h, len(d_vec)))
    
    energy_panneau_vec = np.zeros((len(d_vec), len(betap_vec)))
    energy_radiateur_vec = np.zeros((len(d_vec), len(betap_vec)))
    energy_ecs_vec = np.zeros((len(d_vec), len(betap_vec)))
    energy_circulateur_vec = np.zeros((len(d_vec), len(betap_vec)))
    
    fun_t_ext = fun_temperature_exterieur()

    for i_day, my_day in enumerate(d_vec):
        d_angle = my_day * DAY2RAD
        TEXT_day = fun_t_ext(my_day)
        TCAVE_day = fun_t_ext(my_day)
        is_hiver = TEXT_day < TEXTCHAUFFE
        
        # Initialisation de la puissance totale pour chaque pas de temps
        p_disponible_total[:, i_day] = 0

        for i_panneau in range(len(betap_vec)):
            betap = betap_vec[i_panneau] * DEG2RAD
            thetap = thetap_vec[i_panneau] * DEG2RAD
            eta = eta_vec[i_panneau]
            surface = surface_vec[i_panneau]
            
            for i_hour, hour in enumerate(hvec):
                h = hour * HOUR2RAD + d_angle
                Lst3 = rotation_st_3(d_angle, theta0, h, my_lambda)
                Lst2 = rotation_st_2(d_angle, theta0, h)
                L25 = rotation_2_5(my_lambda, betap, thetap)
                ST5 = (L25.dot(Lst2)).dot(STst)
                ST3 = Lst3.dot(STst)
                unorm = LA.norm(ST5)
                
                tmp1 = ST3[0]
                jour = tmp1 < 0
                
                tmp2 = ST5[0]
                lumiere = tmp2 < 0
                
                if (jour and lumiere):
                    STprojp = np.array([0, ST5[1], ST5[2]])
                    uv = STprojp.dot(ST5)
                    vnorm = LA.norm(STprojp)
                    den = unorm * vnorm
                    if den != 0:
                        my_angle_panneau = np.arccos(uv / den)
                    else:
                        my_angle_panneau = 0.
                else:
                    my_angle_panneau = 0.
                anglevec_p[i_hour, i_day, i_panneau] = my_angle_panneau / DEG2RAD
                
                if jour:
                    STprojl = np.array([0, ST3[1], ST3[2]])
                    uv = STprojl.dot(ST3)
                    vnorm = LA.norm(STprojl)
                    den = unorm * vnorm
                    if den != 0:
                        my_angle_local = np.arccos(uv / den)
                    else:
                        my_angle_local = 0.
                else:
                    my_angle_local = 0.
                
                gamma = np.pi / 2 + my_angle_local
                cg = np.cos(gamma)
                # if cg > 0:
                term1 = R * cg
                term2 = R * np.sqrt(cg**2 + (R2 / R)**2 - 1)
                my_l_atm = term1 + term2
                my_absorption = fun_absorption(my_l_atm)
                # else:
                    # my_absorption = 100
                    # my_l_atm = 0
                
                power_aval = PAMONT * (100 - my_absorption) / 100
                p_aval_vec[i_hour, i_day] = power_aval
                
                ang_tmp = np.pi / 2 - my_angle_panneau
                P_panneau = power_aval * abs(np.cos(ang_tmp))
                
                P_dispo = P_panneau * (eta / 100) * surface
                p_disponible_per_panel[i_hour, i_day, i_panneau] = P_dispo

                # Accumuler la puissance de tous les panneaux
                p_disponible_total[i_hour, i_day] += P_dispo

        # Calculer les énergies pour chaque panneau après la boucle horaire
        for i_panneau in range(len(betap_vec)):
            energy_panneau_vec[i_day, i_panneau] = np.sum(p_disponible_per_panel[:, i_day, i_panneau]) * dt / 1E3
        
        # Calculer les puissances du système thermique
        if is_thermique:
            for i_hour, hour in enumerate(hvec):
                p_circu = P_POMPE_CIRCU
                if i_hour == 0:
                    t_ballon_tampon_total[i_hour, i_day] = 30
                    t_cumulus_total[i_hour, i_day] = 30
                else:
                    circulation = p_disponible_total[i_hour - 1, i_day] > p_circu
                    T_eau_tampon = t_ballon_tampon_total[i_hour - 1, i_day]
                    T_eau_cumulus = t_cumulus_total[i_hour - 1, i_day]
                    P_tampon_fuite = LAMBDA * SBALLON * (T_eau_tampon - TCAVE_day) / EBALLON
                    P_cumulus_fuite = LAMBDA * SBALLON * (T_eau_cumulus - TINT) / EBALLON
                    
                    if circulation:
                        P_tuyaux = LAMBDA * STUYAUX * (T_eau_tampon - TEXT_day) / ETUYAUX
                        P_circulateur = p_circu
                    else:
                        P_tuyaux = 0
                        P_circulateur = 0
                        
                    if is_hiver and T_eau_tampon > TINT:
                        tmp = P0 * ((T_eau_tampon - TINT) / DELTAT0) ** GAMMA
                        P_radiateur = max(0, tmp)
                    else:
                        P_radiateur = 0
                    
                    P_ecs_cumulus = 0
                    P_ecs_tampon = 0
                    if (hour > TIMEECS[0]) and (hour < TIMEECS[1]):
                        DeltaT = T_eau_tampon - T_eau_cumulus
                        P_ecs_cumulus = Cp * DeltaT * (MECS / HOUR2SEC)
                        DeltaT = TINT - T_eau_tampon
                        P_ecs_tampon = Cp * DeltaT * (MECS / HOUR2SEC)

                    P_utile_calc = 0
                    if circulation:
                        P_utile_calc = p_disponible_total[i_hour, i_day] - P_tuyaux - P_tampon_fuite
                    
                    P_ECS = P_utile_calc - P_radiateur
                    
                    p_tampon_fuite_total[i_hour, i_day] = P_tampon_fuite
                    p_radiateur_total[i_hour, i_day] = P_radiateur
                    p_cumulus_total[i_hour, i_day] = P_ecs_cumulus - P_cumulus_fuite
                    p_utile_total[i_hour, i_day] = P_utile_calc
                    p_ecs_total[i_hour, i_day] = P_ECS
                    p_tuyaux_total[i_hour, i_day] = -P_tuyaux
                    p_circulateur_total[i_hour, i_day] = P_circulateur
                    
                    P_tampon_calc = (P_utile_calc - p_tampon_fuite_total[i_hour, i_day] - p_tuyaux_total[i_hour, i_day]) + P_ecs_tampon - P_radiateur
                    p_tampon_total[i_hour, i_day] = P_tampon_calc
                    deltaT_tampon = p_tampon_total[i_hour, i_day] * dt * HOUR2SEC / (Cp * MTAMPON)
                    new_T_tampon = T_eau_tampon + deltaT_tampon
                    t_ballon_tampon_total[i_hour, i_day] = new_T_tampon
                    
                    P_cumulus_calc = P_ecs_cumulus - p_cumulus_total[i_hour, i_day]
                    deltaT_cumulus = P_cumulus_calc * dt * HOUR2SEC / (Cp * MCUMULUS)
                    new_T_cumulus = T_eau_cumulus + deltaT_cumulus
                    t_cumulus_total[i_hour, i_day] = new_T_cumulus
            
        else: # Réinitialiser les paramètres thermiques pour le PV
            p_radiateur_total[:] = 0
            p_cumulus_total[:] = 0
            p_tampon_total[:] = 0
            p_utile_total[:] = 0
            p_ecs_total[:] = 0
            p_tuyaux_total[:] = 0
            p_circulateur_total[:] = 0
            t_ballon_tampon_total[:] = 0
            t_cumulus_total[:] = 0


    return (anglevec_p, p_aval_vec, p_disponible_per_panel, p_disponible_total, t_ballon_tampon_total, t_cumulus_total, p_utile_total, p_tampon_total, 
            p_cumulus_total, p_radiateur_total, p_ecs_total, p_tuyaux_total, p_circulateur_total, energy_panneau_vec, energy_radiateur_vec, 
            energy_ecs_vec, energy_circulateur_vec, d_vec, betap_vec, is_thermique)

# Exécuter la simulation
if 'Cp' not in locals(): # Initialisation des variables thermiques pour le cas PV
    Cp, MTAMPON, MCUMULUS, LAMBDA, SBALLON, EBALLON, TCAVE, TINT = 0, 0, 0, 0, 0, 0, 0, 0
    TEXTCHAUFFE, P0, DELTAT0, GAMMA, P_POMPE_CIRCU, LTUYAUX, ETUYAUX, STUYAUX = 0, 0, 0, 0, 0, 0, 0, 0
    MECS, TIMEECS, is_thermique = 0, [0, 0], st.session_state.system_type == "Solaire thermique"

(anglevec_p, p_aval_vec, p_disponible_per_panel, p_disponible_total, t_ballon_tampon_total, t_cumulus_total, p_utile_total, p_tampon_total, 
 p_cumulus_total, p_radiateur_total, p_ecs_total, p_tuyaux_total, p_circulateur_total, energy_panneau_vec, energy_radiateur_vec, 
 energy_ecs_vec, energy_circulateur_vec, d_vec, betap_vec, is_thermique) = run_simulation(
    betap_vec, thetap_vec, eta_vec, surface_vec, d_vec_input, my_lambda,
    Cp, MTAMPON, MCUMULUS, LAMBDA, SBALLON, EBALLON, TCAVE, TINT, 
    TEXTCHAUFFE, P0, DELTAT0, GAMMA, P_POMPE_CIRCU, LTUYAUX, ETUYAUX, STUYAUX, 
    MECS, TIMEECS, analyse_annee, is_thermique, nb_h  # Pass nb_h to the function
)

# =========================================================================== #
#                   AFFICHAGE DES RÉSULTATS                                   #
# =========================================================================== #
st.markdown("---")
st.header("Résultats de la simulation")

if not analyse_annee:
    jour, mois = day_month(d_vec_input[0])
    st.subheader(f"Résultats pour le {int(jour)} {mois}")

    # Données pour les graphiques journaliers
    df_angle = pd.DataFrame()
    for i in range(len(st.session_state.panels)):
        df_temp = pd.DataFrame({
            "Heure [h]": np.linspace(0, 24, nb_h), # Use nb_h here
            "Angle [deg]": anglevec_p[:, 0, i],
            "Panneau": f"Panneau {i+1}"
        })
        df_angle = pd.concat([df_angle, df_temp], ignore_index=True)
    
    df_power = pd.DataFrame()
    list_df_power = []
    # Créer une ligne pour chaque panneau
    for i in range(len(st.session_state.panels)):
        list_df_power.append(pd.DataFrame({
            "Heure [h]": np.linspace(0, 24, nb_h), # Use nb_h here
            "Puissance [W]": p_disponible_per_panel[:, 0, i],
            "Type": f"Panneau {i+1} (Puissance)"
        }))
    # Ajouter la ligne pour la puissance totale
    list_df_power.append(pd.DataFrame({
        "Heure [h]": np.linspace(0, 24, nb_h), # Use nb_h here
        "Puissance [W]": p_disponible_total[:, 0],
        "Type": "Puissance totale"
    }))
    df_power = pd.concat(list_df_power, ignore_index=True)

    df_power_aval = pd.DataFrame({
        "Heure [h]": np.linspace(0, 24, nb_h), # Use nb_h here
        "Puissance aval [W/m²]": p_aval_vec[:, 0]
    })
    
    df_temp = pd.concat([
        pd.DataFrame({
            "Heure [h]": np.linspace(0, 24, nb_h), # Use nb_h here
            "Température": t_ballon_tampon_total[:, 0],
            "Source": "Tampon"
        }), 
        pd.DataFrame({
            "Heure [h]": np.linspace(0, 24, nb_h), # Use nb_h here
            "Température": t_cumulus_total[:, 0],
            "Source": "Cumulus"
        })
    ])
    
    df_components = pd.DataFrame({
        "Heure [h]": np.linspace(0, 24, nb_h), # Use nb_h here
        "Efficace": p_utile_total[:, 0] / 1E3,
        "Tampon": p_tampon_total[:, 0] / 1E3,
        "Cumulus": p_cumulus_total[:, 0] / 1E3,
        "Radiateur": p_radiateur_total[:, 0] / 1E3,
        "ECS": p_ecs_total[:, 0] / 1E3,
        "Tuyaux": p_tuyaux_total[:, 0] / 1E3
    }).melt("Heure [h]", var_name="Composant", value_name="Puissance [kW]")

    # Graphique 1: Angle panneau (Plotly)
    fig_angle = px.line(df_angle, x="Heure [h]", y="Angle [deg]", color="Panneau",
                        title="Angle des panneaux par rapport au soleil")
    fig_angle.update_yaxes(range=[0, 90])
    st.plotly_chart(fig_angle, use_container_width=True)
    
    # Graphique 2: Puissances (Plotly)
    fig_power = px.line(df_power, x="Heure [h]", y="Puissance [W]", color="Type", 
                        title="Puissance disponible")
    fig_power.update_yaxes(tickformat=", .2f")
    # Récupérer les bornes de l'axe x
    x_min = df_power["Heure [h]"].min()
    x_max = df_power["Heure [h]"].max()
    
    # Ajouter la ligne horizontale
    fig_power.add_trace(
        go.Scatter(
            x=[x_min, x_max],
            y=[2200, 2200],
            mode="lines",
            name=f"Machine à laver: 2200 W",
            line=dict(color="red", dash="dash")
        )
    )
    st.plotly_chart(fig_power, use_container_width=True)
    
    # Graphique 3: Températures (Plotly)
    if is_thermique:
        fig_temp = px.line(df_temp, x="Heure [h]", y="Température", color="Source", 
                            title="Températures des ballons")
        st.plotly_chart(fig_temp, use_container_width=True)

    # Graphique 4: Puissance du système (Plotly)
    if is_thermique:
        fig_components = px.line(df_components, x="Heure [h]", y="Puissance [kW]", color="Composant",
                                     title="Puissance des différents composants")
        st.plotly_chart(fig_components, use_container_width=True)
        
    # Calculate dt outside the if statement for consistency
    dt = (np.linspace(0, 24, nb_h))[1] - (np.linspace(0, 24, nb_h))[0]
    st.write(f"Énergie totale récupérée sur la journée: {np.sum(p_disponible_total)*dt/1e3:.2f} kWh")
    if is_thermique:
        st.write(f"Énergie des radiateurs sur la journée: {np.sum(p_radiateur_total)*dt/1e3:.2f} kWh")
        st.write(f"Énergie ECS sur la journée: {np.sum(p_ecs_total)*dt/1e3:.2f} kWh")
        st.write(f"Énergie totale consommée par la pompe du circulateur sur la journée: {np.sum(p_circulateur_total)*dt/1e3:.2f} kWh")

else:
    # Données pour les graphiques annuels
    d_vec_display = np.arange(0, 365, 1)
    hvec_plot = np.linspace(0, 24, nb_h)
    dt = hvec_plot[1] - hvec_plot[0]

    df_energy = pd.DataFrame()
    list_df = []
    for i_panneau in range(len(betap_vec)):
        df_temp = pd.DataFrame({
            "Jour de l'année": d_vec_display,
            "Énergie totale [kWh]": np.sum(p_disponible_per_panel[:, :, i_panneau] * dt, axis=0) / 1E3,
            "Panneau": f"Panneau {i_panneau+1}"
        })
        list_df.append(df_temp)
    df_energy = pd.concat(list_df, ignore_index=True)
    
    my_duree_jour = np.zeros(365)
    for i_day in range(365):
        try:
            non_zero_indices = np.nonzero(anglevec_p[:, i_day, 0])[0]
            if len(non_zero_indices) > 0:
                my_duree_jour[i_day] = (non_zero_indices.max() - non_zero_indices.min()) * (24 / nb_h)
        except IndexError:
            pass
    df_daylight = pd.DataFrame({
        "Jour de l'année": d_vec_display,
        "Durée [h]": my_duree_jour
    })


    df_thermal_energy = pd.DataFrame()
    if is_thermique:
        list_df_thermal = []
        # Recalculate energy_radiateur_vec and energy_ecs_vec based on the new nb_h
        new_energy_radiateur_vec = np.sum(p_radiateur_total * dt, axis=0) / 1e3
        new_energy_ecs_vec = np.sum(p_ecs_total * dt, axis=0) / 1e3

        df_temp_rad = pd.DataFrame({
            "Jour de l'année": d_vec_display,
            "Énergie [kWh]": new_energy_radiateur_vec,
            "Type": "Radiateur"
        })
        df_temp_ecs = pd.DataFrame({
            "Jour de l'année": d_vec_display,
            "Énergie [kWh]": new_energy_ecs_vec,
            "Type": "ECS"
        })
        list_df_thermal.append(df_temp_rad)
        list_df_thermal.append(df_temp_ecs)
        df_thermal_energy = pd.concat(list_df_thermal, ignore_index=True)

    df_efficiency = pd.DataFrame()
    list_df_efficiency = []
    for i_panneau in range(len(betap_vec)):
        total_energy_per_panel = np.sum(p_disponible_per_panel[:, :, i_panneau] * dt, axis=0) / 1e3
        
        # Gérer la division par zéro
        if np.sum(p_aval_vec) == 0:
            ref_energy = 1
        else:
            ref_energy = np.sum(p_aval_vec * dt, axis=0) * surface_vec[i_panneau] / 1E3
        
        if ref_energy[0] != 0:
            efficacite = (total_energy_per_panel / ref_energy) * 100
        else:
            efficacite = np.zeros(365)
        
        df_temp = pd.DataFrame({
            "Jour de l'année": d_vec_display,
            "Efficacité [%]": efficacite,
            "Panneau": f"Panneau {i_panneau+1}"
        })
        list_df_efficiency.append(df_temp)
    df_efficiency = pd.concat(list_df_efficiency, ignore_index=True)

    # Graphique 1: Énergies (Plotly)
    fig_energy = px.line(df_energy, x="Jour de l'année", y="Énergie totale [kWh]", color="Panneau",
                         title="Énergie cumulée par jour")
    st.plotly_chart(fig_energy, use_container_width=True)

    # Graphique 2: Durée du jour (Plotly)
    fig_daylight = px.line(df_daylight, x="Jour de l'année", y="Durée [h]",
                           title="Durée du jour")
    st.plotly_chart(fig_daylight, use_container_width=True)

    # Graphique 3: Énergie radiateur et ECS (Plotly)
    if is_thermique:
        fig_thermal = px.line(df_thermal_energy, x="Jour de l'année", y="Énergie [kWh]", color="Type",
                              title="Énergie pour radiateur et ECS")
        st.plotly_chart(fig_thermal, use_container_width=True)

    # Graphique 4: Efficacité (Plotly)
    fig_efficiency = px.line(df_efficiency, x="Jour de l'année", y="Efficacité [%]", color="Panneau",
                             title="Efficacité de l'installation")
    st.plotly_chart(fig_efficiency, use_container_width=True)

    st.markdown("---")
    st.subheader("Synthèse annuelle")
    for i_panneau in range(len(betap_vec)):
        st.write(f"**Panneau {i_panneau+1}**")
        st.write(f"Énergie totale récupérée sur l'année: {np.sum(np.sum(p_disponible_per_panel[:, :, i_panneau] * dt, axis=0)) / 1e3:.2f} kWh")
        if is_thermique:
            st.write(f"Énergie totale des radiateurs sur l'année: {np.sum(np.sum(p_radiateur_total * dt, axis=0)) / 1e3:.2f} kWh")
            st.write(f"Énergie totale ECS sur l'année: {np.sum(np.sum(p_ecs_total * dt, axis=0)) / 1e3:.2f} kWh")
            st.write(f"Énergie totale consommée par la pompe du circulateur sur l'année: {np.sum(np.sum(p_circulateur_total * dt, axis=0)) / 1e3:.2f} kWh")











