import pandas as pd

FOREST_EXCEL_PATH = "data/forest_data.xlsx"  # muuta polku jos eri

def load_forest_data():
    """
    Lataa metsämaan kauppahintadata (>10 ha) Excelistä
    """
    df = pd.read_excel(FOREST_EXCEL_PATH)

    # Yhtenäistetään sarakenimet
    df = df.rename(columns={
        "vuosi": "Vuosi",
        "Maakunta": "Maakunta",
        "rakentamattomat, > 10 ha (lukumäärä kpl)": "Kauppojen lukumäärä",
        "rakentamattomat, > 10 ha (mediaani €/ha)": "Mediaani €/ha",
        "rakentamattomat, > 10 ha (keskihinta €/ha)": "Keskihinta €/ha",
        "rakentamattomat, > 10 ha (keskihajonta €/ha)": "Keskihajonta €/ha"
    })

    # Tiputetaan rivit joilta puuttuu oleellinen tieto
    df = df.dropna(subset=["Vuosi", "Maakunta", "Keskihinta €/ha"])

    return df
