import streamlit as st
import folium
from streamlit_folium import st_folium
import datetime
from datetime import timedelta
from branca.element import Template, MacroElement
import pandas as pd
import numpy as np
import altair as alt
import sys
from pathlib import Path
import lightgbm as lgb
import joblib
current_dir = Path.cwd()
project_dir = current_dir.parent
sys.path.append(str(project_dir))
from pipeline import prepare_data, prepare_weather

# RUN COMMAND: python -m streamlit run dashboard.py --server.maxUploadSize 2048          

# Page configuration
st.set_page_config(page_title="Verkeersverwachting Dashboard", layout="wide")


# Color logic config, these are criteria for the colors
# If you change the values here they will for the entire dashboard
GRENS_ORANJE    = 1800     # Kans op file
GRENS_ROOD      = 2200     # Ernstige file
BOVENGRENS      = 3000     # Visualisation maximum

KLEUR_GROEN = '#28a745'
KLEUR_ORANJE = '#ffc107'
KLEUR_ROOD = '#dc3545'

@st.cache_resource
def load_model():
    model_path = Path('model') / 'lightgbm_traffic_model.pkl'
    if model_path.exists():
        return joblib.load(model_path)
    return None

lgbm_model = load_model()


@st.cache_data
def get_cached_data(raw_df):
    """Slaat de geprepareerde data op in het geheugen"""
    return prepare_data(raw_df)


def get_earlier_date(date):
    try:
        year_earlier = date.replace(year=date.year - 1)
    except ValueError:
        # Handle leap year edge case (Feb 29)
        year_earlier = date.replace(year=date.year - 1, day=28)

    current_weekday = date.weekday()
    earlier_weekday = year_earlier.weekday()
    
    day_diff = current_weekday - earlier_weekday
    result = year_earlier + timedelta(days=day_diff)
    
    return result

def get_traffic_data(start_tijd, eind_tijd, datum, locatie, data, intensiteit):
    df = data[data["id_meetlocatie"] == locatie].copy()

    start = datetime.datetime.combine(datum, start_tijd)
    eind = datetime.datetime.combine(datum, eind_tijd)

    df['start_meetperiode'] = pd.to_datetime(df['start_meetperiode'])
    df['date_only'] = df['start_meetperiode'].dt.date
    
    adjusted_datum = datum
    max_iterations = 10  # Safety limit to prevent infinite loop
    iterations = 0
    
    while adjusted_datum not in df['date_only'].values and iterations < max_iterations:
        adjusted_datum = get_earlier_date(adjusted_datum)
        iterations += 1
    
    start = datetime.datetime.combine(adjusted_datum, start_tijd)
    eind = datetime.datetime.combine(adjusted_datum, eind_tijd)

    selected_tijd = df[
        (df["start_meetperiode"] >= start) &
        (df["start_meetperiode"] <= eind)
    ].copy()

    selected_tijd["gem_intensiteit_smooth"] = selected_tijd["gem_intensiteit_smooth"] * (intensiteit/100)

    generated_df = pd.DataFrame({
        'Tijd': selected_tijd["start_meetperiode"].values,
        'Intensiteit': selected_tijd["gem_intensiteit_smooth"].values
    })
    
    return generated_df


def handle_csv_upload():
    st.title("CSV File Upload")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✓ Successfully loaded: {uploaded_file.name}")
            return df
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            return None
    return None


def get_coords(locatie_id, data):
    """
    Gets coördinates of given locaties. Returns None if no geo data.
    """
    if "geometry" not in data.columns:
        return None

    try:
        row = data.loc[data["id_meetlocatie"] == locatie_id]
        if row.empty or pd.isna(row["geometry"].iloc[0]):
            return None
            
        geom = str(row["geometry"].iloc[0])
        parts = geom.replace("POINT", "").replace("(", "").replace(")", "").split()
        return [float(parts[1]), float(parts[0])]
    except Exception:
        return None


# Title
st.title("Verkeersverwachtingen dashboard")

st.write("Kies een dag, tijd, locatie en bepaal toe- of afname in aantal auto's per uur om de verwachtingen op files te zien op de kaart.")

dataframe = handle_csv_upload()
if dataframe is None:
    st.warning("⚠️ Upload een CSV-bestand om de dashboard te gebruiken.")
    st.info("Gebruik de file uploader hierboven om uw verkeersdata te uploaden.")
    st.info("Upload alleen bestanden die hoofdwegen bevatten.")
    st.stop()

# INPUT SECTION
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

# Date input box
with col1:
    st.write("Datum")
    datum = st.date_input("datum", datetime.date.today(), label_visibility="collapsed")

# Time input box
with col2:
    st.write("Tijd")
    tijd_col_a, tijd_col_b = st.columns(2)
    with tijd_col_a:
        start_tijd = st.time_input("Van", datetime.time(8, 0), label_visibility="collapsed")
    with tijd_col_b:
        eind_tijd = st.time_input("Tot", datetime.time(20, 0), label_visibility="collapsed")
   
# Location input box
with col3:
    st.write("Locatie selectie")
    locatie_opties = sorted(dataframe['id_meetlocatie'].unique())
    
    # Initialiseer session_state if non existent
    if 'geselecteerde_locatie' not in st.session_state:
        st.session_state.geselecteerde_locatie = locatie_opties[0]

    # use session state for getting location
    geselecteerde_locatie = st.selectbox(
        "Kies meetpunt", 
        options=locatie_opties, 
        index=locatie_opties.index(st.session_state.geselecteerde_locatie),
        label_visibility="collapsed"
    )
    # Update session
    st.session_state.geselecteerde_locatie = geselecteerde_locatie

# Intensity input box
with col4:
    intensiteit_percentage = st.slider("Auto's per uur", min_value=0, max_value=200, value=100, format="%d%%")  # This is the percentage input
    intensiteit = intensiteit_percentage

st.markdown("---")

# Prepare data
df_prepared = get_cached_data(dataframe)

# Prepare data for visualisation
df = get_traffic_data(start_tijd, eind_tijd, datum, geselecteerde_locatie, df_prepared, intensiteit)

# Get cords from prepared data
geselecteerde_cords = get_coords(geselecteerde_locatie, df_prepared)


if not df.empty:
    gemiddelde_intensiteit = df['Intensiteit'].max()
else:
    gemiddelde_intensiteit = 0
# Dynamic colors for map
if gemiddelde_intensiteit < GRENS_ORANJE:
    icon_color = 'green'
elif GRENS_ORANJE <= gemiddelde_intensiteit < GRENS_ROOD:
    icon_color = 'orange'
else:
    icon_color = 'red'

# OUTPUT SECTION
st.subheader(f"File verwachtingen op: {geselecteerde_locatie}")

# Layout
col_map, col_viz, col_legend = st.columns([1.2, 1.2, 0.5])

# Map
with col_map:
    st.write("**Locatie**")
    
    # Get coördinates
    geselecteerde_cords = get_coords(geselecteerde_locatie, df_prepared)
    
    # Check if there are coördinates
    if geselecteerde_cords is not None:
        # Initialize map with all valid data
        m = folium.Map(location=geselecteerde_cords, zoom_start=14)
        all_points = [geselecteerde_cords] 
        
        folium.Marker(
            location=geselecteerde_cords,
            popup=f"Geselecteerd: {geselecteerde_locatie}",
            icon=folium.Icon(color=icon_color, icon='car', prefix='fa')
        ).add_to(m)
        
        # Add NoN selected locations
        if "geometry" in df_prepared.columns:
            unique_locs = df_prepared[['id_meetlocatie', 'geometry']].drop_duplicates()
            for _, row in unique_locs.iterrows():
                loc_id = row['id_meetlocatie']
                if loc_id != geselecteerde_locatie:
                    other_lat_lon = get_coords(loc_id, df_prepared)
                    if other_lat_lon:
                        all_points.append(other_lat_lon) 
                        folium.CircleMarker(
                            location=other_lat_lon,
                            radius=8,
                            color='blue',
                            fill=True,
                            tooltip=loc_id
                        ).add_to(m)
            m.fit_bounds(all_points)

        # Render map
        map_output = st_folium(
            m, 
            width="100%", 
            height=400, 
            key="verkeers_kaart_vast",
            returned_objects=["last_object_clicked_tooltip"]
        )

        # Click selector
        if map_output and map_output.get("last_object_clicked_tooltip"):
            geklikte_locatie = map_output["last_object_clicked_tooltip"]
            if geklikte_locatie != st.session_state.geselecteerde_locatie:
                st.session_state.geselecteerde_locatie = geklikte_locatie
                st.rerun()
    else:
        # If no shapefile, map not loaded
        st.info(f"📍 **Locatie op de kaart niet gevonden**")
        st.warning(f"""
            **Status van de weergave:**
            De verkeerskundige data voor meetpunt **{geselecteerde_locatie}** is succesvol geladen, 
            maar de bijbehorende geografische coördinaten zijn niet beschikbaar. 
            
            **Mogelijke oorzaak:**
            De koppeling met het bronbestand voor meetlocaties (`Telpunten_WGS84.shp`) kon niet 
            worden gelegd. Controleer of dit bestand geheel aanwezig is in de map:
            `meetpunt_locaties/NDW_AVG_Meetlocaties_Shapefile/`
            
            **Gevolg:**
            De grafieken en voorspellingsmodellen hiernaast zijn volledig operationeel en 
            gebaseerd op de brongegevens en zijn dus nog bruikbaar, de visuele weergave op de kaart is tijdelijk uitgeschakeld.
        """)

# Visualisation on intensity
with col_viz:
    st.write("**Drukte verloop**")
    start_min = start_tijd.hour * 60 + start_tijd.minute
    eind_min = eind_tijd.hour * 60 + eind_tijd.minute

    if eind_min <= start_min:
        st.error("Eindtijd moet na starttijd liggen.")
    else:
        # Dynamic coloring
        df["Status"] = "Groen"

        df.loc[df["Intensiteit"] >= GRENS_ORANJE, "Status"] = "Oranje"
        df.loc[df["Intensiteit"] >= GRENS_ROOD, "Status"] = "Rood"


        # barchart
        bars = alt.Chart(df).mark_bar().encode(
            x=alt.X(
                'Tijd:T',
                timeUnit='yearmonthdatehoursminutes',
                title="Tijdstip",
                axis=alt.Axis(format='%H:%M')
            ),
            y=alt.Y(
                'Intensiteit:Q',
                title="Aantal auto's",
                scale=alt.Scale(domain=[0, BOVENGRENS])
            ),
            color=alt.Color(
                'Status:N',
                scale=alt.Scale(
                    domain=["Groen", "Oranje", "Rood"],
                    range=[KLEUR_GROEN, KLEUR_ORANJE, KLEUR_ROOD]
                ),
                legend=None
            )
        )



        line = alt.Chart(df).mark_line(
            color='white',
            strokeWidth=2
        ).encode(
            x='Tijd:T',
            y='Intensiteit:Q'
        )

        chart = (bars + line).properties(height=400)


        st.altair_chart(chart)


# Legend
with col_legend:
    st.write("**Legenda**")
    st.markdown(f"""
    <div style="font-size: 0.9rem; line-height: 2; border: 1px solid #ccc; padding: 15px; border-radius: 8px;">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="width: 15px; height: 15px; background-color: {KLEUR_GROEN}; margin-right: 10px; border-radius: 2px;"></div>
            <span>Geen file</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="width: 15px; height: 15px; background-color: {KLEUR_ORANJE}; margin-right: 10px; border-radius: 2px;"></div>
            <span>Kans op file</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="width: 15px; height: 15px; background-color: {KLEUR_ROOD}; margin-right: 10px; border-radius: 2px;"></div>
            <span>Ernstige file</span>
        </div>
        <hr style="margin: 10px 0; border: 0; border-top: 1px solid #eee;">
        <div style="display: flex; align-items: center;">
            <div style="width: 15px; height: 15px; background-color: transparent; border: 1px dashed #999; margin-right: 10px; border-radius: 2px;"></div>
            <span style="color: #666; font-style: italic;">Onvoeldoende data beschikbaar</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Stop als model of data niet beschikbaar is
if lgbm_model is None:
    st.info("**ML-model niet beschikbaar.** Train het model en sla het op als een .pkl bestand.")
    st.stop()

if df.empty:
    st.stop()

if 'show_predictions' not in st.session_state:
    st.session_state.show_predictions = False
    
st.write("de Toon ML voorspellingen functionaliteit is conceptueel en de voorspellingen zijn niet bruikbaar in de praktijk.")

# Toon alleen knop als voorspellingen verborgen zijn
if not st.session_state.show_predictions:
    if st.button("Toon ML voorspellingen"):
        st.session_state.show_predictions = True
        st.rerun()
    st.stop()

# Knoppen voor verbergen en model herladen
col_btn1, col_btn2 = st.columns([3, 1])
with col_btn1:
    if st.button("Verberg ML voorspellingen"):
        st.session_state.show_predictions = False
        st.rerun()

with col_btn2:
    if st.button("Herlaad Model"):
        st.cache_resource.clear()
        st.rerun()
        
col_ml_metrics, col_ml_zoom, col_spacer = st.columns([1.2, 1.2, 0.5])

with col_ml_metrics:
    st.write("**Model voorspelling**")

    # Voeg features toe
    df_pred = df.copy()
    df_pred['tijd_sec'] = df_pred['Tijd'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
    df_pred['day_of_week'] = datum.weekday()

    # Maak DataFrame met juiste kolomnamen voor het model
    # Het model verwacht deze exacte kolomnamen uit de training data
    X_dashboard = df_pred[['Intensiteit', 'tijd_sec', 'day_of_week']].copy()
    X_dashboard.columns = ['gem_intensiteit_smooth', 'tijd', 'day_of_week']

    try:
        # Voorspel kans op omslagpunt (0/1) voor elk tijdstip
        df_pred['kans_omslagpunt'] = lgbm_model.predict_proba(X_dashboard)[:, 1]

        # Markeer punten met >30% kans als waarschuwing
        # Deze threshold is gekozen omdat het model lage precisie (3%) maar hoge recall (77%) heeft
        df_pred['is_waarschuwing'] = df_pred['kans_omslagpunt'] > 0.3

        max_kans = df_pred['kans_omslagpunt'].max()
        n_warnings = int(df_pred['is_waarschuwing'].sum())
        
        st.metric("Hoogste risico", f"{max_kans:.1%}")
        st.metric("Gedetecteerde omslagpunten", f"{n_warnings}")

        if n_warnings > 0:
            # Vind tijdstip met hoogste risico
            hoogste_idx = int(df_pred['kans_omslagpunt'].idxmax())
            hoogste_tijd_val = df_pred.loc[hoogste_idx, 'Tijd']
            try:
                tijd_str = hoogste_tijd_val.strftime('%H:%M')
            except AttributeError:
                tijd_str = str(hoogste_tijd_val)
            st.metric("Hoogste risico tijdstip", tijd_str)

            if max_kans > 0.3:
                st.error("**Omslagpunt gedetecteerd**")
            else:
                st.success("**Laag risico**")
        else:
            st.success("**Geen waarschuwingen**")
            hoogste_idx = 0

    except Exception as e:
        st.warning(f"Model fout: {str(e)}")
        df_pred = None
        n_warnings = 0
        hoogste_idx = 0

with col_ml_zoom:
    st.write("**Omslagpunt Detectie**")

    if df_pred is not None and n_warnings > 0:
        df_window = df_pred.copy()
        df_window['Punt_type'] = 'Normaal'
        df_window.loc[df_window['is_waarschuwing'], 'Punt_type'] = 'Omslagpunt'

        # Staafdiagram
        zoom_bars = alt.Chart(df_window).mark_bar(opacity=0.7).encode(
        x=alt.X(
            'Tijd:T',
            title='Tijd',
            axis=alt.Axis(format='%H:%M')),
        y=alt.Y('Intensiteit:Q',
                title="Auto's/uur"),
        color=alt.condition(
            alt.datum.Punt_type == 'Omslagpunt',
            alt.value('#dc3545'),
            alt.value('#6c757d')
            ))

        zoom_line = alt.Chart(df_window).mark_line(
            color='black',
            strokeWidth=2,
            point=True
        ).encode(
            x='Tijd:T',
            y='Intensiteit:Q'
        )
        
        omslagpunt_marker = alt.Chart(df_window[df_window['Punt_type'] == 'Omslagpunt']).mark_point(
            size=400,
            color='red',
            shape='triangle-down',
            filled=True
        ).encode(
            x='Tijd:T',
            y='Intensiteit:Q',
            tooltip=[
                alt.Tooltip('Tijd:T', title='Omslagpunt', format='%H:%M'),
                alt.Tooltip('Intensiteit:Q', title='Intensiteit', format='.0f'),
                alt.Tooltip('kans_omslagpunt:Q', title='Kans', format='.1%')
            ]
        )

        zoom_chart = (zoom_bars + zoom_line + omslagpunt_marker).properties(
            height=300,
            title="Dit punt voorspelt een file binnen 25 minuten"
        )

        st.altair_chart(zoom_chart, use_container_width=True)
        st.caption("Rood driehoekje = Omslagpunt (steilste daling voor file)")
    else:
        st.info("Geen omslagpunten gedetecteerd in deze periode")
