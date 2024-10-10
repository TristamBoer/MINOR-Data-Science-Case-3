import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def page_config():
    st.set_page_config(layout='wide')


page_config()

st.sidebar.success('')

st.title('Laadpalen & Elektrisch vervoer Dashboard')
st.markdown(
    """
    Binnen deze Dashboard wordt er gekeken naar verschillende ondervindingen van de Laadpalen & Elektrisch vervoer datasets.  
    Hierin worden de volgende onderdelen besproken:
    - Aantal cumulatieve voertuigen met een Lineair Regression model.  
    - Verdeling van Laadpunten in Nederland.  
    - Gemiddelde elektriciteit prijs voor laadpunten.  
    - Populaire merken en modellen.  
    - Laadpunten per stad in een folium map.
    """
)

@st.cache_data
def df():
    return pd.read_pickle('cars2.pkl')


cars_df = df()

# Functie gemaakt door Nathan Isaac (team 19), met toestemming gebruikt.
def bepaal_brandstof(naam):
    """ Assigns a type of fuel to different cars based on vehicle trade name
    
    Args:
        naam (str): Trade name of the vehicle
        
    Returns:
        (str)
    """

    naam = naam.lower()
    if any(keyword in naam for keyword in
           ['edrive', 'id', 'ev', 'electric', 'atto', 'pro', 'ex', 'model', 'e-tron', 'mach-e', 'kw']):
        return 'elektrisch'
    elif any(keyword in naam for keyword in ['hybrid', 'phev', 'plugin']):
        return 'hybride'
    elif 'diesel' in naam:
        return 'diesel'
    elif 'waterstof' in naam or 'fuel cell' in naam:
        return 'waterstof'
    else:
        return 'benzine'


cars_df['brandstof_omschrijving'] = cars_df['handelsbenaming'].apply(bepaal_brandstof)
cars_df['datum_tenaamstelling_dt'] = pd.to_datetime(cars_df['datum_tenaamstelling_dt'], errors='coerce')
cars_df['maand_jaar'] = cars_df['datum_tenaamstelling_dt'].dt.to_period('M')
grouped = cars_df.groupby(['maand_jaar', 'brandstof_omschrijving']).size().reset_index(name='Aantal_voertuigen')
grouped['Cumulatief_aantal_voertuigen'] = grouped.groupby('brandstof_omschrijving')['Aantal_voertuigen'].cumsum()
grouped['maand_jaar'] = grouped['maand_jaar'].astype(str)

column1, column2 = st.columns(2)

with column1:
    fig = px.line(
        grouped,
        x='maand_jaar', y='Cumulatief_aantal_voertuigen',
        title='Lijndiagram cumulatieve aantal voertuigen per maand en type brandstof',
        color='brandstof_omschrijving',
        color_discrete_map={
            'elektrisch': 'red',
            'benzine': 'blue'
        }
    )

    fig.update_layout(
        {
            'xaxis': {'title': {'text': 'Maand'}},
            'yaxis': {'title': {'text': 'Aantal voertuigen'}},
            'legend_title': ''
        }
    )

    st.plotly_chart(fig)

with column2:
    fig = px.pie(
        grouped,
        values='Aantal_voertuigen',
        names='brandstof_omschrijving',
        title="Pie chart van de verhouding tussen benzine en elektrische auto's",
        color='brandstof_omschrijving',
        color_discrete_map={
            'elektrisch': 'red',
            'benzine': 'blue'
        }
    )

    st.plotly_chart(fig)

def numpy_lineair_regression(dataframe, x_axis, y_axis, parameters=False, prediction=False, n=0):
    """ Numpy based lineair regression prediction model

    Args:
        dataframe (Pandas DataFrame): DataFrame containing data needed for prediction.
        x_axis (Pandas DataFrame column): X-axis of predicted values.
        y_axis = (Pandas DataFrame column): Y-axis of predicted values.
        parameters (bool, optional): Prints slope and intercept.
        prediction (bool, optional): Checks if function needs to predict values.
        n (int): Decides how much the function predicts.

    Returns:
        If parameters=True:
            list with slope and intercept values
        If prediction=True:
            Pandas DataFrame columns with the predicted values
        Else:
            Lineair Regression prediction function
    """
    x = np.arange(dataframe[x_axis].size)
    fit = np.polyfit(x, dataframe[y_axis], deg=1)
    fit_function = np.poly1d(fit)

    if parameters:
        return [fit[0], fit[1]]

    if prediction:
        return fit_function(dataframe[x_axis].size + np.arange(n))

    return fit_function(x)


fig = go.Figure()

df = grouped[grouped['brandstof_omschrijving'] == 'benzine']

fig.add_trace(go.Scatter(
    x=df['maand_jaar'],
    y=df['Cumulatief_aantal_voertuigen'],
    mode='lines+markers',
    name='Aantal benzine voertuigen',
    marker_color='Blue'
))

fig.add_trace(go.Scatter(
    x=df['maand_jaar'],
    y=numpy_lineair_regression(df, 'maand_jaar', 'Cumulatief_aantal_voertuigen'),
    mode='lines',
    name='Benzine lineair model',
    marker_color='aqua'
))

df = grouped[grouped['brandstof_omschrijving'] == 'elektrisch']

fig.add_trace(go.Scatter(
    x=df['maand_jaar'],
    y=df['Cumulatief_aantal_voertuigen'],
    mode='lines+markers',
    name='Aantal elektrische voertuigen',
    marker_color='red'
))

fig.add_trace(go.Scatter(
    x=df['maand_jaar'],
    y=numpy_lineair_regression(df, 'maand_jaar', 'Cumulatief_aantal_voertuigen'),
    mode='lines',
    name='Elektrische llineair model',
    marker_color='pink'
))

fig.update_layout(
    title="Lineair voorspellend model aantal voertuigen",
    xaxis_title="Datum",
    yaxis_title="Aantal cumulatieve voertuigen",
)

st.plotly_chart(fig)

@st.cache_data
def API_shivano():
    url = "https://api.openchargemap.io/v3/poi"
    api_key = "a887fc1e-bb00-417f-9dc2-be020b34d5d1"
    country_code = "NL"  # Landcode voor Nederland
    max_results = 7957  # Maximum aantal resultaten per aanroep

    return requests.get(url, params={
        'key': api_key,
        'countrycode': country_code,
        'maxresults': max_results
    })


# Maak een DataFrame van de laadpunten
df = pd.DataFrame(API_shivano().json())

# Groepeer op provincie en tel het aantal laadpunten
province_counts = df['AddressInfo'].apply(lambda x: x.get('StateOrProvince')).value_counts()

# Combineer verschillende namen voor dezelfde provincies
combined_provinces = {
    'Utrecht': ['Utrecht', 'UT', 'UTRECHT'],
    'Gelderland': ['Gelderland'],
    'Noord-Holland': ['North Holland', 'Noord-Holland', 'North-Holland', 'Noord Holand'],
    'Zuid-Holland': ['Zuid-Holland', 'Zuid Holland', 'South Holland', 'ZH'],
    'Zeeland': ['Zeeland', 'Seeland'],
    'Noord-Brabant': ['Noord-Brabant', 'North Brabant', 'Noord Brabant'],
    'Overijssel': ['Overijssel'],
    'Flevoland': ['Flevoland'],
    'Limburg': ['Limburg'],
    'Groningen': ['Groningen'],
    'Drenthe': ['Drenthe'],
    'Friesland': ['Friesland', 'Fryslân', 'FRL']
}

# Tel de laadpunten per provincie
final_counts = {}
for province, names in combined_provinces.items():
    final_counts[province] = sum(province_counts.get(name, 0) for name in names)

# Maak een DataFrame voor de final_counts
final_df = pd.DataFrame(list(final_counts.items()), columns=['Province', 'Charging Points'])
final_df = final_df.sort_values(by='Charging Points', ascending=False)

# Plot de gegevens met Plotly in een enkele groene kleur
fig = px.bar(final_df, x='Province', y='Charging Points',
             title='Verdeling van Laadpunten per Provincie',
             labels={'Province': 'Provincie', 'Charging Points': 'Aantal Laadpunten'},
             )

fig.update_layout(xaxis_title='Provincie', yaxis_title='Aantal Laadpunten', xaxis_tickangle=-45)
st.plotly_chart(fig)


@st.cache_data
def csv_file():
    return pd.read_csv('2013tm2023.csv')


df_jaren = csv_file()


month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
               7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

df_jaren['month'] = df_jaren['month'].map(month_names)


def create_histogram(x_axis_column):
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df_jaren[x_axis_column],
        y=df_jaren['Prijs'],
        histfunc='avg',
    ))

    fig.update_layout(
        title=f'Histogram of prijs per {x_axis_column.capitalize()}',
        xaxis_title=x_axis_column.capitalize(),
        yaxis_title='Gemiddelde prijs kWh [€]',
        xaxis=dict(tickangle=45 if x_axis_column == 'month' else 0),
        bargap=0.2,
    )

    return fig


column1, column2 = st.columns([1, 5])

with column1:
    x_axis_option = st.radio('Selecteer x-as:', ('year', 'month', 'hour'))

with column2:
    fig = create_histogram(x_axis_option)
    st.plotly_chart(fig)


@st.cache_data
def rwd_api():
    return pd.read_json("https://opendata.rdw.nl/resource/m9d7-ebf2.json?$limit=100000")


RDWvoertuigen_df = rwd_api()

# Omzetten van datum naar datetime-formaat
RDWvoertuigen_df['datum_eerste_tenaamstelling_in_nederland'] = pd.to_datetime(
    RDWvoertuigen_df['datum_eerste_tenaamstelling_in_nederland'], format='%Y%m%d')
# set 'date' column to proper format, filter a new column holding the year and numeric month.
RDWvoertuigen_df['year'] = RDWvoertuigen_df['datum_eerste_tenaamstelling_in_nederland'].dt.year
# print(RDWvoertuigen_df['year'].value_counts())

non_cars = ['Middenasaanhangwagen', 'Motorfiets', 'Aanhangwagen', 'Bromfiets', 'Oplegger', 'Driewielig motorrijtuig',
            'Autonome aanhangwagen', 'Land- of bosb aanhw of getr uitr stuk', 'Mobiele machine',
            'Motorfiets met zijspan']
RDWvoertuigen_df.drop(RDWvoertuigen_df.loc[RDWvoertuigen_df['voertuigsoort'].isin(non_cars)].index, inplace=True)
RDWvoertuigen_df['aantal_cilinders'] = RDWvoertuigen_df['aantal_cilinders'].fillna(0)

elektrische_merken = RDWvoertuigen_df[RDWvoertuigen_df['aantal_cilinders'] == 0]

# Streamlit page title
st.title("EV popularity via registration count from RDW datasets")

# Selectbox for choosing plot type
plot_type = st.selectbox("Kies een plot type:", ["20 meest geregistreerde merken", "20 meest geregistreerde modellen"])

# plots data for brand choice
if plot_type == "20 meest geregistreerde merken":
    # set up variable for most popular brands from RDW and the average prices for each these brands.
    populaire_merken = elektrische_merken['merk'].value_counts().head(20)
    average_prices = elektrische_merken.groupby('merk')['catalogusprijs'].mean().loc[populaire_merken.index]

    # Plots bars for brand count
    fig, ax1 = plt.subplots(figsize=(10, 5))
    populaire_merken.plot(kind='bar', color='skyblue', ax=ax1, position=0, width=0.4)
    ax1.set_title('20 meest populaire elektrische merken')
    ax1.set_xlabel('Merk', fontsize=10)
    ax1.set_ylabel('Aantal registraties', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.grid(axis='y')

    # Plots bars for brand average price
    ax2 = ax1.twinx()
    average_prices.plot(kind='bar', color='orange', ax=ax2, position=1, width=0.4)
    ax2.set_ylabel('Gemiddelde catalogusprijs', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    st.pyplot(fig)

# plots data for model choice
elif plot_type == "20 meest geregistreerde modellen":
    # set up variable for most popular models from RDW and the average prices for each these models.
    populaire_modellen = elektrische_merken['handelsbenaming'].value_counts().head(20)
    average_model_prices = elektrische_merken.groupby('handelsbenaming')['catalogusprijs'].mean().loc[
        populaire_modellen.index]

    # Plots bars for model count
    fig, ax1 = plt.subplots(figsize=(10, 5))
    populaire_modellen.plot(kind='bar', color='lightgreen', ax=ax1, position=0, width=0.4)
    ax1.set_title('Populaire modellen elektrische voertuigen')
    ax1.set_xlabel('Model', fontsize=10)
    ax1.set_ylabel('Aantal voertuigen', color='lightgreen')
    ax1.tick_params(axis='y', labelcolor='lightgreen')
    ax1.grid(axis='y')

    # Plots bars for model average price
    ax2 = ax1.twinx()
    average_model_prices.plot(kind='bar', color='red', ax=ax2, position=1, width=0.4)
    ax2.set_ylabel('Gemiddelde catalogusprijs', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    st.pyplot(fig)
