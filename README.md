# Team Panda - Traffic & More

Een interactief Streamlit dashboard voor verkeersanalyse op Nederlandse hoofdwegen. Test eenvoudig verschillende scenario's door de verkeersintensiteit aan te passen en zie direct de impact op file-verwachtingen.

## Overzicht

Dit project biedt een visueel dashboard waarmee je verkeersintensiteit kunt analyseren en verschillende scenario's kunt simuleren. De kernfunctionaliteit is de **intensiteit slider** waarmee je direct kunt zien wat er gebeurt bij meer of minder verkeer.

### Belangrijkste Functionaliteiten

- **Intensiteit Slider (0-200%)**: De hoofdfunctie - pas verkeersintensiteit aan en zie direct de impact
- **Interactieve Kaart**: Visualiseer alle meetlocaties op een Folium kaart met kleurcodering
- **Klikbare Locaties**: Klik op een meetpunt op de kaart om deze te selecteren
- **Tijdserie Grafiek**: Zie verkeersintensiteit over de dag met bar chart + line overlay
- **Realtime Kleurcodering**: Groen (normaal) → Oranje (kans op file) → Rood (ernstige file)
- **Historische Data Matching**: Als een datum niet beschikbaar is, zoekt het systeem automatisch een vergelijkbare dag een jaar eerder

- ⚠️ **ML Voorspellingen (WIP)**: Omslagpunt detectie (experimenteel, nog niet accuraat)


## Project Structuur

```
TeamPanda-TrafficAndMore/
├── dashboard.py                  # Hoofdapplicatie (Streamlit dashboard)
├── pipeline.py                   # Data preprocessing pipelines
├── requirements.txt              # Python dependencies
├── model/
│   ├── train_model.py           # Training script voor LightGBM model
│   └── lightgbm_traffic_model.pkl  # Getraind ML model (optioneel)
├── meetpunt_locaties/
│   └── NDW_AVG_Meetlocaties_Shapefile/
│       ├── Telpunten_WGS84.shp  # Shapefile met meetpunt locaties
│       ├── Telpunten_WGS84.shx
│       ├── Telpunten_WGS84.dbf
│       ├── Telpunten_WGS84.prj
│       ├── Meetvakken_WGS84.*   # Bijbehorende shapefile bestanden
│       └── ...
└── references/
    ├── data_preparatie.ipynb     # Notebook voor data exploratie
    └── definitie_kantelpunt.ipynb # Notebook voor omslagpunt definitie
```

## Installatie

### Vereisten

- Python 3.10 of hoger

### Stappen

1. **Clone of download het project**
   ```bash
   git clone https://github.com/JustMyla/TeamPanda-TrafficAndMore.git 
   cd TeamPanda-TrafficAndMore
   ```

2. **Installeer dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start het dashboard**
   ```bash
   python -m streamlit run dashboard.py --server.maxUploadSize 2048
   ```

   Het dashboard opent automatisch in je browser op `http://localhost:8501`