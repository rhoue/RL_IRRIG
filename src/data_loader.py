"""
Module pour charger des données réelles (CSV ou API) pour l'irrigation intelligente.

Ce module permet d'intégrer :
- Données de pluie réelles (CSV ou API météo)
- Données de tension matricielle réelles (CSV ou API capteurs)
- Données d'ET0 réelles (CSV ou API météo)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import requests
from datetime import datetime, timedelta


# ============================================================================
# CHARGEMENT DEPUIS CSV
# ============================================================================

def load_weather_from_csv(
    file_path: str,
    date_col: str = "date",
    rain_col: str = "rain",
    et0_col: Optional[str] = "et0",
    date_format: str = "%Y-%m-%d"
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Charge les données météorologiques depuis un fichier CSV.
    
    FORMAT CSV ATTENDU :
    date,rain,et0
    2024-01-01,0.0,3.5
    2024-01-02,5.2,4.1
    ...
    
    Args:
        file_path (str): Chemin vers le fichier CSV
        date_col (str): Nom de la colonne contenant les dates
        rain_col (str): Nom de la colonne contenant la pluie (mm)
        et0_col (str, optional): Nom de la colonne contenant l'ET0 (mm/j)
        date_format (str): Format des dates (par défaut "%Y-%m-%d")
        
    Returns:
        tuple: (rain, et0, Kc) où :
            - rain (np.ndarray): Pluie journalière (mm)
            - et0 (np.ndarray ou None): ET0 journalière (mm/j) si disponible
            - Kc (np.ndarray): Coefficient cultural (calculé si ET0 manquant)
    """
    df = pd.read_csv(file_path)
    
    # Conversion des dates
    df[date_col] = pd.to_datetime(df[date_col], format=date_format)
    df = df.sort_values(date_col)
    
    # Extraction de la pluie
    rain = df[rain_col].values.astype(np.float32)
    
    # Extraction de l'ET0 si disponible
    if et0_col and et0_col in df.columns:
        et0 = df[et0_col].values.astype(np.float32)
    else:
        et0 = None
    
    # Calcul du Kc (coefficient cultural) basé sur le jour de l'année
    # Approche simplifiée : évolution typique d'une culture
    days_since_start = np.arange(len(df))
    Kc = np.zeros(len(df), dtype=np.float32)
    for t in days_since_start:
        if t < 20:
            Kc[t] = 0.3
        elif t < 50:
            Kc[t] = 0.3 + (1.15 - 0.3) * (t - 20) / (50 - 20)
        elif t < 90:
            Kc[t] = 1.15
        else:
            Kc[t] = 1.15 + (0.7 - 1.15) * (t - 90) / max(len(df) - 90, 1)
    
    return rain, et0, Kc


def load_tension_from_csv(
    file_path: str,
    date_col: str = "date",
    tension_col: str = "tension",
    date_format: str = "%Y-%m-%d"
) -> np.ndarray:
    """
    Charge les données de tension matricielle depuis un fichier CSV.
    
    FORMAT CSV ATTENDU :
    date,tension
    2024-01-01,35.2
    2024-01-02,42.1
    ...
    
    Args:
        file_path (str): Chemin vers le fichier CSV
        date_col (str): Nom de la colonne contenant les dates
        tension_col (str): Nom de la colonne contenant la tension (cbar)
        date_format (str): Format des dates
        
    Returns:
        np.ndarray: Tensions matricielles (cbar)
    """
    df = pd.read_csv(file_path)
    df[date_col] = pd.to_datetime(df[date_col], format=date_format)
    df = df.sort_values(date_col)
    
    tension = df[tension_col].values.astype(np.float32)
    return tension


# ============================================================================
# CHARGEMENT DEPUIS API
# ============================================================================

def load_weather_from_api(
    api_url: str,
    start_date: str,
    end_date: str,
    api_key: Optional[str] = None,
    params: Optional[Dict] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Charge les données météorologiques depuis une API.
    
    EXEMPLE D'UTILISATION :
    # API OpenWeatherMap (nécessite une clé API)
    rain, et0, Kc = load_weather_from_api(
        api_url="https://api.openweathermap.org/data/2.5/forecast",
        start_date="2024-01-01",
        end_date="2024-03-31",
        api_key="votre_cle_api",
        params={"lat": 45.5, "lon": 2.3, "units": "metric"}
    )
    
    Args:
        api_url (str): URL de l'API
        start_date (str): Date de début (format "YYYY-MM-DD")
        end_date (str): Date de fin (format "YYYY-MM-DD")
        api_key (str, optional): Clé API si nécessaire
        params (dict, optional): Paramètres supplémentaires pour l'API
        
    Returns:
        tuple: (rain, et0, Kc)
    """
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        # Ou selon l'API : params["api_key"] = api_key
    
    # Calcul du nombre de jours
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    num_days = (end - start).days + 1
    
    # Appel API (à adapter selon l'API utilisée)
    # Exemple générique :
    if params:
        response = requests.get(api_url, headers=headers, params=params)
    else:
        response = requests.get(api_url, headers=headers)
    
    response.raise_for_status()
    data = response.json()
    
    # Extraction des données (à adapter selon le format de réponse de l'API)
    # Exemple pour une API qui retourne une liste de jours :
    rain = np.zeros(num_days, dtype=np.float32)
    et0 = np.zeros(num_days, dtype=np.float32) if "et0" in str(data) else None
    
    # TODO: Adapter selon le format de réponse de votre API
    # Par exemple :
    # for i, day_data in enumerate(data["daily"]):
    #     rain[i] = day_data.get("rain", 0.0)
    #     if et0 is not None:
    #         et0[i] = calculate_et0_from_api_data(day_data)
    
    # Calcul du Kc
    days_since_start = np.arange(num_days)
    Kc = np.zeros(num_days, dtype=np.float32)
    for t in days_since_start:
        if t < 20:
            Kc[t] = 0.3
        elif t < 50:
            Kc[t] = 0.3 + (1.15 - 0.3) * (t - 20) / (50 - 20)
        elif t < 90:
            Kc[t] = 1.15
        else:
            Kc[t] = 1.15 + (0.7 - 1.15) * (t - 90) / max(num_days - 90, 1)
    
    return rain, et0, Kc


def load_tension_from_api(
    api_url: str,
    start_date: str,
    end_date: str,
    sensor_id: Optional[str] = None,
    api_key: Optional[str] = None
) -> np.ndarray:
    """
    Charge les données de tension depuis une API de capteurs.
    
    EXEMPLE D'UTILISATION :
    # API de capteurs IoT
    tension = load_tension_from_api(
        api_url="https://api.capteurs.com/v1/measurements",
        start_date="2024-01-01",
        end_date="2024-03-31",
        sensor_id="sensor_123",
        api_key="votre_cle_api"
    )
    
    Args:
        api_url (str): URL de l'API
        start_date (str): Date de début
        end_date (str): Date de fin
        sensor_id (str, optional): ID du capteur
        api_key (str, optional): Clé API
        
    Returns:
        np.ndarray: Tensions matricielles (cbar)
    """
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    params = {
        "start_date": start_date,
        "end_date": end_date
    }
    if sensor_id:
        params["sensor_id"] = sensor_id
    
    response = requests.get(api_url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    
    # Extraction (à adapter selon le format de réponse)
    # Exemple :
    # tensions = [measurement["tension"] for measurement in data["measurements"]]
    # return np.array(tensions, dtype=np.float32)
    
    # Placeholder
    num_days = (datetime.strptime(end_date, "%Y-%m-%d") - 
                datetime.strptime(start_date, "%Y-%m-%d")).days + 1
    return np.zeros(num_days, dtype=np.float32)


# ============================================================================
# FONCTION UNIFIÉE POUR STREAMLIT
# ============================================================================

def load_data_for_simulation(
    data_source: str = "synthetic",
    file_path: Optional[str] = None,
    api_url: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Fonction unifiée pour charger des données depuis différentes sources.
    
    Args:
        data_source (str): "synthetic", "csv", ou "api"
        file_path (str, optional): Chemin vers le fichier CSV
        api_url (str, optional): URL de l'API
        start_date (str, optional): Date de début (format "YYYY-MM-DD")
        end_date (str, optional): Date de fin (format "YYYY-MM-DD")
        **kwargs: Paramètres supplémentaires (api_key, sensor_id, etc.)
        
    Returns:
        dict: Dictionnaire avec les clés :
            - "rain": pluie (mm)
            - "et0": ET0 (mm/j) si disponible
            - "Kc": coefficient cultural
            - "tension": tensions (cbar) si disponibles
    """
    if data_source == "csv":
        if not file_path:
            raise ValueError("file_path requis pour data_source='csv'")
        
        # Chargement météo
        rain, et0, Kc = load_weather_from_csv(file_path, **kwargs)
        
        # Chargement tensions si fichier séparé
        tension_file = kwargs.get("tension_file")
        tension = None
        if tension_file:
            tension = load_tension_from_csv(tension_file, **kwargs)
        
        result = {
            "rain": rain,
            "Kc": Kc
        }
        if et0 is not None:
            result["et0"] = et0
        if tension is not None:
            result["tension"] = tension
        
        return result
    
    elif data_source == "api":
        if not api_url or not start_date or not end_date:
            raise ValueError("api_url, start_date et end_date requis pour data_source='api'")
        
        # Chargement météo
        rain, et0, Kc = load_weather_from_api(
            api_url, start_date, end_date, **kwargs
        )
        
        # Chargement tensions si API séparée
        tension_api = kwargs.get("tension_api_url")
        tension = None
        if tension_api:
            tension = load_tension_from_api(
                tension_api, start_date, end_date, **kwargs
            )
        
        result = {
            "rain": rain,
            "Kc": Kc
        }
        if et0 is not None:
            result["et0"] = et0
        if tension is not None:
            result["tension"] = tension
        
        return result
    
    else:  # synthetic
        # Retourner None pour indiquer qu'on utilise generate_weather
        return None


