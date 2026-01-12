from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.signal import savgol_filter
from pathlib import Path
import os

# look up path
BASE_DIR = Path(__file__).resolve().parent
shapefile_pad = BASE_DIR / "meetpunt_locaties" / "NDW_AVG_Meetlocaties_Shapefile"
shapefile_file = shapefile_pad / "Telpunten_WGS84.shp"

# Initialize at none
telpunten_gdf = None

if shapefile_file.exists():
    try:
        telpunten_gdf = gpd.read_file(str(shapefile_file))
    except Exception as e:
        print(f"Fout bij het laden van shapefile: {e}")
else:
    print(f"Waarschuwing: Shapefile niet gevonden op {shapefile_file}. De 'merge_with_geo' stap zal worden overgeslagen.")


def drop_columns(X):
    if 'traffic_flow_deviation_exclusions' in X.columns:
        X = X.drop(columns="traffic_flow_deviation_exclusions")
    return X


def remove_missing(X):
    if 'technical_exclusion' in X.columns:
        X = X.drop(columns=['technical_exclusion'])
    
    if 'gem_snelheid' in X.columns:
        X = X[X.gem_snelheid.notna()]
        
    return X


def remove_negs(X): 
    if 'gem_snelheid' in X.columns:
        X = X[X.gem_snelheid != -1]
    
    if 'gem_intensiteit' in X.columns:
        X = X[X.gem_intensiteit != -1]
    
    return X


def drop_low(X):
    if 'gem_intensiteit' in X.columns:
        X = X[X.gem_intensiteit > 50]
    return X


def datetime(X):
    if 'start_meetperiode' not in X.columns:
        return X
    
    X['start_meetperiode'] = pd.to_datetime(X['start_meetperiode'], errors='coerce')
    X['datum'] = X['start_meetperiode'].dt.date
    return X


def time_to_int(X):
    if 'start_meetperiode' not in X.columns:
        return X
    
    # Ensure it's datetime
    if not pd.api.types.is_datetime64_any_dtype(X['start_meetperiode']):
        X['start_meetperiode'] = pd.to_datetime(X['start_meetperiode'], errors='coerce')
    
    X['tijd'] = (X['start_meetperiode'].dt.hour * 3600 + 
                 X['start_meetperiode'].dt.minute * 60 + 
                 X['start_meetperiode'].dt.second)
    return X


def lanes(X):
    if 'rijstrook_rijbaan' not in X.columns:
        return X
    return X[X['rijstrook_rijbaan'] == 'lane1']


def dayofweek(X):
    if 'datum' not in X.columns:
        return X
    
    X["day_of_week"] = pd.to_datetime(X["datum"], errors='coerce').dt.dayofweek
    return X


def merge_with_geo(X):
    # no merge if there's no geodata
    if telpunten_gdf is None:
        print("Slaan merge_with_geo over: telpunten_gdf is niet beschikbaar.")
        return X
    
    # prepare for merge
    geo_info = telpunten_gdf[['dgl_loc', 'wegtype', 'geometry']].copy()
    geo_info.rename(columns={'dgl_loc': 'id_meetlocatie'}, inplace=True)

    # merge
    data_f_verrijkt = pd.merge(X, geo_info, on='id_meetlocatie', how='left')

    # convert to gdf
    gdf_verrijkt = gpd.GeoDataFrame(
        data_f_verrijkt.dropna(subset=['geometry']),
        geometry='geometry',
        crs="EPSG:4326"
    )
    return gdf_verrijkt


def smooth_data(X):
    y = X['gem_snelheid'].values
    y_smooth = savgol_filter(y, window_length=min(7, len(y) - (1 if len(y) % 2 == 0 else 0)), polyorder=2)
    X["gem_snelheid_smooth"] = y_smooth

    z = X['gem_intensiteit'].values
    z_smooth = savgol_filter(z, window_length=min(7, len(z) - (1 if len(z) % 2 == 0 else 0)), polyorder=2)
    X["gem_intensiteit_smooth"] = z_smooth   
    
    return X


def add_helling_per_punt(df, window_size=5, slope_col="helling_per_punt"):
    df = df.copy()

    if "id_meetlocatie" not in df.columns:
        raise ValueError("Kolom 'id_meetlocatie' ontbreekt in de DataFrame.")

    # sorteren per meetlocatie en tijd
    df = df.sort_values(["id_meetlocatie", "start_meetperiode"]).reset_index(drop=True)

    # lege kolom voor hellingen
    df[slope_col] = np.nan

    # per meetlocatie apart helling berekenen
    for loc_id, group in df.groupby("id_meetlocatie", sort=False):
        idx = group.index.to_list()
        speeds = group["gem_snelheid"].values

        slopes = np.full(len(group), np.nan)

        for i in range(len(group) - window_size + 1):
            window = speeds[i:i + window_size]

            # alleen fitten wanneer alle waarden geldig zijn
            if np.all(np.isfinite(window)):
                x = np.arange(window_size)
                slope, _ = np.polyfit(x, window, 1)
                slopes[i] = slope

        # terugschrijven in originele df
        df.loc[idx, slope_col] = slopes

    return df

# van dit punt zijn er een aantal functies die niet van ons zijn
def detect_files(df, start_speed=50, end_speed=75, end_steps_required=2):
    """
    Detecteer files per meetlocatie en ken een file-ID toe per segment.

    Logica:
    - File start wanneer snelheid < start_speed
    - File stopt pas nadat end_steps_required opeenvolgende punten >= end_speed zijn
    """

    df = df.copy()

    # We moeten per locatie aparte file-id's bepalen
    if "id_meetlocatie" not in df.columns:
        raise ValueError("Kolom 'id_meetlocatie' ontbreekt.")

    df = df.sort_values(["id_meetlocatie", "start_meetperiode"]).reset_index(drop=True)

    df["file_id"] = np.nan
    current_file = 0  # numerieke teller voor file-segmenten

    # Per meetlocatie apart analyseren
    for loc_id, group in df.groupby("id_meetlocatie", sort=False):
        idx = group.index
        speeds = group["gem_snelheid"].values

        inside_file = False
        end_streak = 0  # hoeveel punten op rij boven herstel-waarde

        # Itereer chronologisch door rij-indexen van deze locatie
        for k, row_idx in enumerate(idx):
            speed = speeds[k]

            # Wanneer we niet in een file zitten → wacht op daling
            if not inside_file:
                if speed < start_speed:
                    inside_file = True
                    current_file += 1  # nieuwe file start
                    df.at[row_idx, "file_id"] = current_file
                    end_streak = 0
            else:
                # We zitten binnen een file → zelfde ID blijft lopen
                df.at[row_idx, "file_id"] = current_file

                # Check herstelconditie
                if speed >= end_speed:
                    end_streak += 1
                    # wanneer snel genoeg herstel is gedetecteerd → file eindigt
                    if end_streak >= end_steps_required:
                        inside_file = False
                        end_streak = 0
                else:
                    # herstel is onderbroken
                    end_streak = 0

    print(f"Totaal gedetecteerde files: {int(df['file_id'].nunique(dropna=True))}")
    return df


def mark_omslagpunten(df, window_back=5, slope_col="helling_per_punt"):
    """
    Per file (per meetlocatie) wordt één omslagpunt gemarkeerd.
    """

    df = df.copy()

    if "id_meetlocatie" not in df.columns:
        raise ValueError("Kolom 'id_meetlocatie' niet gevonden.")
    if "file_id" not in df.columns:
        raise ValueError("Kolom 'file_id' niet gevonden. Draai eerst detect_files(df).")
    if slope_col not in df.columns:
        raise ValueError(
            f"Kolom '{slope_col}' niet gevonden. "
            "Draai eerst add_helling_per_punt(df) of geef een andere slope_col mee."
        )

    # sorteren per meetlocatie én tijd, zodat indexblokken per locatie netjes zijn
    df = df.sort_values(["id_meetlocatie", "start_meetperiode"]).reset_index(drop=True)

    df["file_omslag_flag"] = 0

    n_files = int(df["file_id"].nunique(dropna=True))
    n_marked = 0

    # per meetlocatie apart omslagpunten zoeken
    for loc_id, group in df.groupby("id_meetlocatie", sort=False):
        loc_idx = group.index
        loc_min_idx = loc_idx.min()

        # voor deze locatie: laatste omslagindex (mag niet overlappen)
        last_omslag_idx = loc_min_idx - 1

        files_loc = group["file_id"].dropna().unique()

        for fid in files_loc:
            file_idx = group.index[group["file_id"] == fid].tolist()
            if not file_idx:
                continue

            start_idx = file_idx[0]

            # window vóór file-start, maar binnen deze meetlocatie blijven
            start_window = max(loc_min_idx, start_idx - window_back)
            end_window = start_idx - 1
            if end_window < start_window:
                continue

            window_idx = list(range(start_window, end_window + 1))

            # laatste index in window met file_id (eerdere file binnen dezelfde locatie)
            file_indices_in_window = [
                i for i in window_idx if not pd.isna(df.at[i, "file_id"])
            ]
            last_file_idx_in_window = max(file_indices_in_window) if file_indices_in_window else -1

            # grens: na laatste omslagpunt én na laatste file-index in window
            threshold = max(last_omslag_idx, last_file_idx_in_window)

            # kandidaten: zonder file_id en ná threshold
            candidate_idx = [
                i for i in window_idx
                if pd.isna(df.at[i, "file_id"]) and i > threshold
            ]

            omslag_idx = None

            if candidate_idx:
                slopes_window = df.loc[candidate_idx, slope_col].dropna()
                if not slopes_window.empty:
                    negatives = slopes_window[slopes_window < 0]
                    if not negatives.empty:
                        omslag_idx = negatives.idxmin()      # meest negatieve helling
                    else:
                        omslag_idx = slopes_window.idxmin()  # kleinste (beste) helling

            # fallback: geen kandidaten → punt direct vóór file-start
            if omslag_idx is None:
                if end_window > last_omslag_idx:
                    omslag_idx = end_window
                else:
                    continue  # zelfs dat niet mogelijk

            df.at[omslag_idx, "file_omslag_flag"] = 1
            n_marked += 1
            last_omslag_idx = omslag_idx

    print("-----------------------------------------------------")
    print(f"Detected files (unique file_id): {n_files}")
    print(f"Marked omslagpunten:             {n_marked}")
    print("-----------------------------------------------------")

    return df


missing_dropper = FunctionTransformer(remove_missing)
neg_dropper = FunctionTransformer(remove_negs)
low_dropper = FunctionTransformer(drop_low)
time_fixer = FunctionTransformer(datetime)
time_converter = FunctionTransformer(time_to_int)
data_smoother = FunctionTransformer(smooth_data)
day_adder = FunctionTransformer(dayofweek)
geo_merger = FunctionTransformer(merge_with_geo)
column_dropper = FunctionTransformer(drop_columns)
lane_selector = FunctionTransformer(lanes)



pipeline = Pipeline(steps=[
    ('drop_missing_values', missing_dropper),
    ('remove_negative_values', neg_dropper),
    ('drop_low_values', low_dropper),
    ('fix_date_and_time', time_fixer),
    ('convert time to int', time_converter),
    ('smooth_data', data_smoother),
    ("add day of week", day_adder),
    ('merge_geo', geo_merger),
    ('drop_columns', column_dropper),
    ("select lanes", lane_selector)
])


def prepare_data(data):
    return pipeline.fit_transform(data)

helling_adder = FunctionTransformer(add_helling_per_punt)
file_detector = FunctionTransformer(detect_files)
omslagpunter_marker = FunctionTransformer(mark_omslagpunten)

pipeline_omslagpunten = Pipeline(steps=[
    ("add helling", helling_adder),
    ("detecteer file", file_detector), 
    ("markeer omslagpunten", omslagpunter_marker)
])

def markeer_omslagpunten(data):
    return pipeline_omslagpunten.fit_transform(data)

def fix_names(X):
    X.rename(columns={'YYYYMMDD': 'datum'})
    X.rename(columns={'HH': 'Tijd'})
    X.rename(columns={'FF': 'Wind_snelheid'})
    X.rename(columns={'FX': 'Hoogste_wind_stoot'})
    X.rename(columns={'T': 'Temperatuur'})
    X.rename(columns={'T10N': 'Min_temperatuur'})
    X.rename(columns={'DR': 'Duur_neerslag'})
    X.rename(columns={'RH': 'Uursom_neerslag'})
    X.rename(columns={'VV': 'Horziontaal_zicht'})
    X.rename(columns={'M': 'Mist'})
    X.rename(columns={'R': 'Regen'})
    X.rename(columns={'S': 'Sneeuw'})
    X.rename(columns={'O': 'Onweer'})
    X.rename(columns={'Y': 'IJsvorming'})
    return X


def fix_datetime(X):
    X['datum'] = pd.to_datetime(X['datum'], format='%Y%m%d')
    X['Tijd'] = pd.to_timedelta(X['tijd'], unit='h')
    X['datetime_local'] = X['datetime_utc'].dt.tz_localize('UTC').dt.tz_convert('Europe/Amsterdam')
    return X


name_fixer = FunctionTransformer(fix_names)
datetime_fixer = FunctionTransformer(fix_datetime)


pipeline_weather = Pipeline(steps=[
    ('fix_names', name_fixer),
    ('fix_datetime', datetime_fixer)
])


def prepare_weather(data):
    return pipeline_weather.fit_transform(data)

# place this at your import to make the pipeline importable and working
# import sys
# from pathlib import Path
# current_dir = Path.cwd()
# project_dir = current_dir.parent
# sys.path.append(str(project_dir))
# from Data_Preperatie.pipeline import prepare_data, prepare_weather

def undersample(df, minority_class=1, ratio=3, random=17):
    df_minority = df[df["file_omslag_flag"] == minority_class].copy()
    df_majority = df[df["file_omslag_flag"] != minority_class].copy()
    n_minority = len(df_minority)
    n_majority_target = int(n_minority * ratio)
    df_majority_undersampled = df_majority.sample(n=n_majority_target,random_state=random)
    df_undersampled = pd.concat([df_minority, df_majority_undersampled], ignore_index=True)    
    return df_undersampled

undersampeler = FunctionTransformer(undersample)

pipeline_sampling = Pipeline(steps=[
    ("undersample data",undersampeler )
])
