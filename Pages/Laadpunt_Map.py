import streamlit as st
from streamlit_folium import st_folium

import pandas as pd
import folium
import requests
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def page_config():
    st.set_page_config(layout='wide')


page_config()

st.sidebar.success('')


def get_api_data(url, limit=None):
    """ Gets values from online API

    Args:
        url (str): url to the API
        limit (int, optional): Sets the limit to the amount of imported data from API

    Returns:
        API data in JSON format.
    """

    if limit is not None:
        url = f'{url}?$limit={limit}'
    else:
        url = url

    response = requests.get(url)
    data = response.json()
    return data


def get_api_df(data):
    """ Converts API JSON data to Pandas DataFrame

    Args:
        data (JSON): dataset formatted to JSON

    Returns:
        DataFrame
    """

    df = pd.DataFrame(data)
    return df


api_data = get_api_data(
    "https://api.openchargemap.io/v3/poi/?output=json&countrycode=NL&maxresults=7957&compact=true&verbose=false&key=93b912b5-9d70-4b1f-960b-fb80a4c9c017")
api_df = get_api_df(api_data)
api_df = api_df.drop(
    columns=['GeneralComments', 'OperatorsReference', 'DataProvidersReference', 'MetadataValues', 'DateLastConfirmed'])
api_df['NumberOfPoints'] = api_df['NumberOfPoints'].ffill()

def town_coordinates(stad):
    """ Gets the coordinates for each city

    Args:
        stad (str): name of the city

    Returns:
        coordinates
    """

    if stad == 'Utrecht':
        return [52.0921, 5.1187]
    elif stad == 'Amsterdam':
        return [52.3676, 4.9041]
    elif stad == 'Haarlem':
        return [52.3812, 4.6336]
    elif stad == 'Den Haag':
        return [52.080329, 4.30965]
    elif stad == 'Rotterdam':
        return [51.9228958, 4.4631727]
    elif stad == 'Eindhoven':
        return [51.434619, 5.486011]
    elif stad == 'Arnhem':
        return [51.9851034, 5.8987296]


def color_n_points(n):
    """ Assigns color to the amount of charging points

    Args:
        n (int): amount of charging points

    Returns:
        valid folium color (str)
    """

    if n == 1:
        return 'black'
    elif n == 2:
        return 'gray'
    elif n == 3:
        return 'cadetblue'
    elif n == 4:
        return 'pink'
    elif n == 5:
        return 'red'
    elif n == 6:
        return 'orange'
    elif n == 8:
        return 'purple'
    elif n == 10:
        return 'green'
    elif n == 12:
        return 'blue'
    else:
        return 'beige'


def add_categorical_legend(folium_map, title, colors, labels):
    if len(colors) != len(labels):
        raise ValueError("colors and labels must have the same length.")

    color_by_label = dict(zip(labels, colors))

    legend_categories = ""
    for label, color in color_by_label.items():
        legend_categories += f"<li><span style='background:{color}'></span>{label}</li>"

    legend_html = f"""
    <div id='maplegend' class='maplegend'>
      <div class='legend-title'>{title}</div>
      <div class='legend-scale'>
        <ul class='legend-labels'>
        {legend_categories}
        </ul>
      </div>
    </div>
    """
    script = f"""
        <script type="text/javascript">
        var oneTimeExecution = (function() {{
                    var executed = false;
                    return function() {{
                        if (!executed) {{
                             var checkExist = setInterval(function() {{
                                       if ((document.getElementsByClassName('leaflet-top leaflet-right').length) || (!executed)) {{
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.display = "flex"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.flexDirection = "column"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].innerHTML += `{legend_html}`;
                                          clearInterval(checkExist);
                                          executed = true;
                                       }}
                                    }}, 100);
                        }}
                    }};
                }})();
        oneTimeExecution()
        </script>
      """

    css = """

    <style type='text/css'>
      .maplegend {
        z-index:9999;
        float:right;
        background-color: rgba(255, 255, 255, 1);
        border-radius: 5px;
        border: 2px solid #bbb;
        padding: 10px;
        font-size:12px;
        positon: relative;
      }
      .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
      .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
      .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
      .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 0px solid #ccc;
        }
      .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
      .maplegend a {
        color: #777;
        }
    </style>
    """

    folium_map.get_root().header.add_child(folium.Element(script + css))

    return folium_map


api_location_df = pd.json_normalize(api_df['AddressInfo'])  # JSON data --> Dataframe
api_location_df = pd.concat([api_location_df, api_df[['NumberOfPoints']]], axis=1)

st.title("Laadpaal Locator")
column1, column2 = st.columns(2)

with column1:
    town = st.selectbox("Selecteer een stad:",
                        ['Amsterdam', 'Utrecht', 'Haarlem', 'Den Haag', 'Rotterdam', 'Eindhoven', 'Arnhem'])

with column2:
    selected_n_points = st.radio(
        "Selecteer aantal zichtbare laadpunten:",
        options=['All', 1, 2, 3, 4, 5, 6, 8, 10, 12, 24],
        index=0  # Default selectie is 'All'
    )

def laadpaal_map(town, n_points):
    """ Creates a folium map of different cities in the combined_df DataFrame. With markers assigned to different charging points.

    Args:
        Town (str): The town/city the map & markers will be shown

    Returns:
        folium map
    """

    df = api_location_df[api_location_df['Town'] == town]

    if selected_n_points != 'All':
        df = df[df['NumberOfPoints'] == n_points]

    coordinates = town_coordinates(town)

    m = folium.Map(
        location=coordinates,
        zoom_start=11,
    )

    for i in df.index:
        n_points = df.loc[i, 'NumberOfPoints']

        folium.Marker(
            location=[df.loc[i, 'Latitude'], df.loc[i, 'Longitude']],
            icon=folium.Icon(color=color_n_points(n_points)),
            tooltip='<b>Klik hier om de popup te zien</b>',
            popup=f"Address: {df.loc[i, 'AddressLine1']}, Aantal laadpunten: {n_points}KW"
        ).add_to(m)

    m = add_categorical_legend(m, 'Aantal Laadpunten',
                               colors=['black', 'gray', 'cadetblue', 'pink', 'red', 'orange', 'purple', 'green', 'blue',
                                       'beige'],
                               labels=['1', '2', '3', '4', '5', '6', '8', '10', '12', '24'])

    return m


map_display = laadpaal_map(town, selected_n_points)
st_folium(map_display, width=750)
