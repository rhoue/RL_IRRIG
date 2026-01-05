"""
Utilitaires pour la génération de données météorologiques.

Ce module centralise les fonctions de génération de données météorologiques
utilisées dans les simulations d'irrigation.
"""

import numpy as np


def generate_weather(
    T=120,
    seed=123,
    et0_base=4.0,
    et0_amp=2.0,
    et0_noise=0.3,
    p_rain_early=0.25,
    p_rain_mid=0.15,
    p_rain_late=0.20,
    rain_min=3.0,
    rain_max=25.0
):
    """
    Génère des séries temporelles synthétiques de données météorologiques pour une saison.
    
    PRINCIPE :
    Simulation réaliste de données météo pour une saison culturale :
    - ET0 : évapotranspiration de référence (variation saisonnière + variabilité journalière)
    - Pluie : événements aléatoires avec probabilité variable selon la phase de saison
    - Kc : coefficient cultural (évolution selon le stade de développement de la culture)
    
    MODÈLE ET0 :
    - Signal sinusoïdal : modélise la variation saisonnière (été plus chaud = ET0 plus élevé)
    - Bruit gaussien : ajoute de la variabilité journalière (météo changeante)
    - Clipping : maintient des valeurs réalistes (0.5-8.0 mm/j)
    
    MODÈLE PLUIE :
    - Probabilité variable : plus de pluie en début/fin de saison (printemps/automne)
    - Intensité uniforme : si pluie, intensité entre rain_min et rain_max
    - Événements discrets : pas de pluie continue, mais des épisodes
    
    MODÈLE Kc (coefficient cultural) :
    Représente la demande en eau de la culture selon son stade :
    - Phase initiale (0-20j) : Kc = 0.3 (faible demande, semis/levée)
    - Phase croissance (20-50j) : montée linéaire 0.3 → 1.15 (développement végétatif)
    - Phase développement (50-90j) : Kc = 1.15 (demande maximale, floraison/fructification)
    - Phase maturation (90j-fin) : descente 1.15 → 0.7 (maturation, sénescence)
    
    Args:
        T (int): Longueur de la saison en jours
        seed (int): Graine pour la reproductibilité
        et0_base (float): Valeur de base de l'ET0 (mm/j)
        et0_amp (float): Amplitude de la variation saisonnière (mm/j)
        et0_noise (float): Écart-type du bruit gaussien (mm/j)
        p_rain_early (float): Probabilité de pluie en début de saison (0-1)
        p_rain_mid (float): Probabilité de pluie en milieu de saison (0-1)
        p_rain_late (float): Probabilité de pluie en fin de saison (0-1)
        rain_min (float): Intensité minimale de pluie (mm)
        rain_max (float): Intensité maximale de pluie (mm)
        
    Returns:
        tuple: (rain, et0, Kc) où :
            - rain (np.ndarray): Pluie journalière (mm), shape (T,)
            - et0 (np.ndarray): ET0 journalière (mm/j), shape (T,)
            - Kc (np.ndarray): Coefficient cultural journalier (-), shape (T,)
    """
    # Initialisation du générateur aléatoire pour reproductibilité
    rng = np.random.RandomState(seed)
    days = np.arange(T)

    # ET0: variation saisonnière sinusoïdale + bruit gaussien
    # Modélise l'évolution de l'évapotranspiration de référence au cours de la saison
    # Signal sinusoïdal sur une période = T jours (une saison complète)
    et0 = et0_base + et0_amp * np.sin(2 * np.pi * (days / max(T, 1)))
    # Ajout de bruit gaussien pour variabilité journalière (météo changeante)
    et0 += rng.normal(0.0, et0_noise, size=T)
    # Clipping pour rester dans des valeurs réalistes (évite valeurs négatives ou trop élevées)
    et0 = np.clip(et0, 0.5, 8.0).astype(np.float32)

    # Pluie: événements aléatoires avec probabilité variable selon la phase de saison
    # Probabilité plus élevée en début et fin de saison (printemps/automne)
    rain = np.zeros(T, dtype=np.float32)
    for t in range(T):
        # Probabilité de pluie selon la phase de saison
        if t < T * 0.3:      # Début de saison (30% premiers jours)
            p_rain = p_rain_early
        elif t < T * 0.7:    # Milieu de saison (40% jours suivants)
            p_rain = p_rain_mid
        else:                # Fin de saison (30% derniers jours)
            p_rain = p_rain_late
        # Si pluie, intensité uniforme entre rain_min et rain_max
        rain[t] = rng.uniform(rain_min, rain_max) if rng.rand() < p_rain else 0.0

    # Kc (coefficient cultural): évolution typique d'une culture
    # Représente la demande en eau de la culture selon son stade de développement
    Kc = np.zeros(T, dtype=np.float32)
    for t in range(T):
        if t < 20:
            # Phase initiale : faible demande (Kc = 0.3)
            # Semis, levée, faible couverture du sol
            Kc[t] = 0.3
        elif t < 50:
            # Phase de croissance : montée linéaire de 0.3 à 1.15
            # Développement végétatif, augmentation de la surface foliaire
            Kc[t] = 0.3 + (1.15 - 0.3) * (t - 20) / (50 - 20)
        elif t < 90:
            # Phase de développement maximal : plateau à 1.15
            # Floraison, fructification, demande maximale en eau
            Kc[t] = 1.15
        else:
            # Phase de maturation : descente linéaire vers 0.7
            # Maturation, sénescence, réduction de la demande
            Kc[t] = 1.15 + (0.7 - 1.15) * (t - 90) / max(T - 90, 1)
    return rain, et0, Kc

