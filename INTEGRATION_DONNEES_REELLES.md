# Guide d'intégration de données réelles

Ce guide explique comment intégrer des données réelles (CSV ou API) pour les tensions matricielles et la pluie dans l'application d'irrigation intelligente.

## 📋 Table des matières

1. [Architecture générale](#architecture-générale)
2. [Intégration CSV](#intégration-csv)
3. [Intégration API](#intégration-api)
4. [Modifications du code principal](#modifications-du-code-principal)
5. [Exemples d'utilisation](#exemples-dutilisation)

---

## 🏗️ Architecture générale

### Points d'intégration

Le code actuel utilise :
- `generate_weather()` : génère rain, et0, Kc de manière synthétique
- Modèle physique : calcule les tensions ψ à partir de la réserve S

Pour intégrer des données réelles, il faut modifier :
1. **`IrrigationEnvPhysical.__init__()`** : accepter des données externes au lieu d'appeler `generate_weather()`
2. **`simulate_scenario1()`** : accepter des données externes
3. **Interface Streamlit** : ajouter des options pour charger CSV/API

### Options d'intégration

**Option 1 : Données de pluie réelles uniquement**
- Utiliser les pluies réelles
- Continuer à calculer les tensions via le modèle physique
- Continuer à générer ET0 (ou utiliser données réelles si disponibles)

**Option 2 : Données de tension réelles uniquement**
- Utiliser les tensions mesurées directement
- Continuer à utiliser les pluies générées ou réelles
- Le modèle physique peut être utilisé pour valider/calibrer

**Option 3 : Données complètes (pluie + tension)**
- Utiliser toutes les données réelles
- Le modèle physique sert uniquement à la validation

---

## 📁 Intégration CSV

Les fichiers d'exemple sont disponibles dans le dossier `data/` :
- `data/example_meteo.csv` : Format pour les données météorologiques
- `data/example_tension.csv` : Format pour les données de tension matricielle

### Format CSV attendu

#### Fichier météo (pluie + ET0)

```csv
date,rain,et0
2024-01-01,0.0,3.5
2024-01-02,5.2,4.1
2024-01-03,0.0,4.3
...
```

#### Fichier tensions

```csv
date,tension
2024-01-01,35.2
2024-01-02,42.1
2024-01-03,38.5
...
```

### Utilisation dans Streamlit

```python
# Dans la sidebar, ajouter une section "Source de données"
data_source = st.radio(
    "Source de données",
    options=["Synthétique", "CSV", "API"],
    index=0
)

if data_source == "CSV":
    uploaded_file = st.file_uploader(
        "Charger fichier CSV météo",
        type=["csv"],
        help="Format attendu : date,rain,et0"
    )
    
    if uploaded_file:
        # Charger les données
        from src.data_loader import load_weather_from_csv
        rain, et0, Kc = load_weather_from_csv(uploaded_file)
        
        # Utiliser ces données au lieu de generate_weather()
```

---

## 🌐 Intégration API

### Exemples d'APIs

#### 1. API Météo (OpenWeatherMap)

```python
import requests

def get_weather_from_openweathermap(lat, lon, start_date, end_date, api_key):
    """
    Récupère les données météo depuis OpenWeatherMap.
    Note: L'API gratuite ne donne que les prévisions, pas l'historique.
    Pour l'historique, il faut l'API payante.
    """
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric"
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    # Extraire pluie et calculer ET0
    rain = []
    for item in data["list"]:
        rain.append(item.get("rain", {}).get("3h", 0.0) / 3.0)  # mm/j
    
    return np.array(rain, dtype=np.float32)
```

#### 2. API Capteurs IoT (exemple générique)

```python
def get_tension_from_sensor_api(sensor_id, start_date, end_date, api_key):
    """
    Récupère les tensions depuis une API de capteurs IoT.
    """
    url = f"https://api.capteurs.com/v1/measurements"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "sensor_id": sensor_id,
        "start_date": start_date,
        "end_date": end_date
    }
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    
    tensions = [m["tension"] for m in data["measurements"]]
    return np.array(tensions, dtype=np.float32)
```

---

## 🔧 Modifications du code principal

### 1. Modifier `IrrigationEnvPhysical`

```python
class IrrigationEnvPhysical(gym.Env):
    def __init__(
        self,
        season_length=120,
        max_irrigation=20.0,
        seed=0,
        soil_params: Optional[Dict[str, float]] = None,
        weather_params: Optional[Dict[str, Any]] = None,
        # NOUVEAU : accepter des données externes
        external_rain: Optional[np.ndarray] = None,
        external_et0: Optional[np.ndarray] = None,
        external_Kc: Optional[np.ndarray] = None,
        external_tension: Optional[np.ndarray] = None
    ):
        # ... code existant ...
        
        # MODIFICATION : utiliser données externes si disponibles
        if external_rain is not None:
            # Utiliser données réelles
            self.rain = external_rain.astype(np.float32)
            self.et0 = external_et0.astype(np.float32) if external_et0 is not None else None
            self.Kc = external_Kc.astype(np.float32) if external_Kc is not None else self._generate_Kc(season_length)
            
            # Si tensions réelles fournies, les utiliser directement
            if external_tension is not None:
                self.use_real_tension = True
                self.real_tension = external_tension.astype(np.float32)
            else:
                self.use_real_tension = False
        else:
            # Comportement par défaut : génération synthétique
            weather_kwargs = weather_params if weather_params else {}
            self.rng = np.random.default_rng(seed)
            self.rain, self.et0, self.Kc = generate_weather(
                T=season_length, seed=seed, **weather_kwargs
            )
            self.use_real_tension = False
    
    def step(self, action):
        # ... code existant ...
        
        # MODIFICATION : si tensions réelles, les utiliser
        if self.use_real_tension and self.day < len(self.real_tension):
            psi_next = float(self.real_tension[self.day])
            # Calculer S correspondant pour cohérence
            S_next = float(self.soil.psi_to_S(psi_next))
        else:
            # Comportement normal : calcul via modèle physique
            S_next = np.clip(
                self.S + self.soil.eta_I * action + rain_t - ETc - D,
                0.0, self.soil.S_max
            )
            psi_next = float(self.soil.S_to_psi(S_next))
        
        # ... reste du code ...
```

### 2. Modifier `simulate_scenario1`

```python
def simulate_scenario1(
    T=120,
    seed=0,
    I_max=20.0,
    soil: Optional[PhysicalBucket] = None,
    rule_fn=rule_seuil_unique,
    rule_kwargs=None,
    weather_params: Optional[Dict[str, Any]] = None,
    # NOUVEAU : données externes
    external_rain: Optional[np.ndarray] = None,
    external_et0: Optional[np.ndarray] = None,
    external_tension: Optional[np.ndarray] = None
):
    # ... code existant ...
    
    # MODIFICATION : utiliser données externes si disponibles
    if external_rain is not None:
        rain = external_rain.astype(np.float32)
        et0 = external_et0.astype(np.float32) if external_et0 is not None else None
        # Générer Kc si nécessaire
        Kc = np.zeros(T, dtype=np.float32)
        for t in range(T):
            if t < 20:
                Kc[t] = 0.3
            elif t < 50:
                Kc[t] = 0.3 + (1.15 - 0.3) * (t - 20) / (50 - 20)
            elif t < 90:
                Kc[t] = 1.15
            else:
                Kc[t] = 1.15 + (0.7 - 1.15) * (t - 90) / max(T - 90, 1)
    else:
        # Comportement par défaut
        weather_kwargs = weather_params if weather_params else {}
        rain, et0, Kc = generate_weather(T=T, seed=seed, **weather_kwargs)
    
    # ... reste du code ...
    
    # Si tensions réelles, les utiliser
    if external_tension is not None:
        for t in range(T):
            # Utiliser tension réelle
            psi[t] = float(external_tension[t])
            # Calculer S correspondant
            S[t] = float(soil.psi_to_S(psi[t]))
            # ... reste de la logique ...
```

### 3. Ajouter interface dans Streamlit

```python
# Dans la sidebar
st.markdown("### 📊 Source de données")

data_source = st.radio(
    "Source des données",
    options=["Synthétique", "CSV", "API"],
    index=0,
    help="Choisissez la source des données météorologiques"
)

external_rain = None
external_et0 = None
external_tension = None

if data_source == "CSV":
    st.markdown("#### Charger depuis CSV")
    
    # Fichier météo
    weather_file = st.file_uploader(
        "Fichier CSV météo (date, rain, et0)",
        type=["csv"],
        help="Format : date,rain,et0"
    )
    
    if weather_file:
        from src.data_loader import load_weather_from_csv
        external_rain, external_et0, Kc = load_weather_from_csv(weather_file)
        st.success(f"✅ {len(external_rain)} jours chargés")
    
    # Fichier tensions (optionnel)
    tension_file = st.file_uploader(
        "Fichier CSV tensions (date, tension)",
        type=["csv"],
        help="Format : date,tension"
    )
    
    if tension_file:
        from src.data_loader import load_tension_from_csv
        external_tension = load_tension_from_csv(tension_file)
        st.success(f"✅ {len(external_tension)} mesures chargées")

elif data_source == "API":
    st.markdown("#### Configuration API")
    
    api_type = st.selectbox(
        "Type d'API",
        options=["Météo", "Capteurs IoT"]
    )
    
    if api_type == "Météo":
        api_url = st.text_input("URL API météo")
        api_key = st.text_input("Clé API", type="password")
        start_date = st.date_input("Date de début")
        end_date = st.date_input("Date de fin")
        
        if st.button("Charger données météo"):
            from src.data_loader import load_weather_from_api
            external_rain, external_et0, Kc = load_weather_from_api(
                api_url, str(start_date), str(end_date), api_key=api_key
            )
            st.success(f"✅ {len(external_rain)} jours chargés")
    
    # ... configuration API capteurs ...

# Passer les données aux fonctions
if external_rain is not None:
    # Modifier les appels à simulate_scenario1 et IrrigationEnvPhysical
    sim_result = simulate_scenario1(
        T=len(external_rain),
        seed=seed,
        I_max=max_irrigation,
        soil=soil,
        rule_fn=rule_fn,
        rule_kwargs=rule_kwargs,
        external_rain=external_rain,
        external_et0=external_et0,
        external_tension=external_tension
    )
```

---

## 📝 Exemples d'utilisation

### Exemple 1 : CSV simple

```python
# Créer un fichier CSV
import pandas as pd

data = {
    "date": pd.date_range("2024-01-01", periods=120, freq="D"),
    "rain": np.random.exponential(2.0, 120),
    "et0": 4.0 + 2.0 * np.sin(np.arange(120) * 2 * np.pi / 120)
}
df = pd.DataFrame(data)
df.to_csv("data/meteo_2024.csv", index=False)

# Charger dans Streamlit
from src.data_loader import load_weather_from_csv
rain, et0, Kc = load_weather_from_csv("data/meteo_2024.csv")
```

**Fichiers d'exemple disponibles :**
- `data/example_meteo.csv` : Exemple de données météorologiques
- `data/example_tension.csv` : Exemple de données de tension matricielle

### Exemple 2 : API OpenWeatherMap

```python
# Nécessite : pip install requests
import requests

def get_weather_openweathermap(lat, lon, api_key):
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric"
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    # Extraire pluie
    rain = []
    for item in data["list"]:
        rain_3h = item.get("rain", {}).get("3h", 0.0)
        rain.append(rain_3h / 3.0)  # Convertir en mm/j
    
    return np.array(rain, dtype=np.float32)
```

---

## ⚠️ Points d'attention

1. **Alignement temporel** : S'assurer que les données de pluie, ET0 et tensions sont alignées sur les mêmes dates
2. **Longueur des séries** : Vérifier que les données couvrent toute la saison
3. **Unités** : Vérifier les unités (mm pour pluie, cbar pour tension, mm/j pour ET0)
4. **Valeurs manquantes** : Gérer les NaN avec interpolation ou valeurs par défaut
5. **Validation** : Comparer les tensions réelles avec celles calculées par le modèle pour valider

---

## 🔄 Prochaines étapes

1. Implémenter les modifications dans `IrrigationEnvPhysical`
2. Ajouter l'interface dans Streamlit
3. Tester avec les fichiers d'exemple dans `data/` :
   - `data/example_meteo.csv` pour les données météorologiques
   - `data/example_tension.csv` pour les données de tension
4. Intégrer une API réelle (selon vos besoins)
5. Ajouter la gestion des erreurs et validation

