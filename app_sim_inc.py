import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import geopandas as gpd
import time
import datetime as dt
from shapely.ops import cascaded_union
from shapely.geometry import Point
import numpy as np
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt
import random
import folium
import time
import plotly.express as px
from streamlit_folium import st_folium, folium_static
from PIL import Image
import os
import sqlalchemy as sa
from sqlalchemy import select, and_, func
import oracledb

### Libreria de simulaciones
import lib.sim_model_library as sim_model

################################ SET PAGE
st.set_page_config(
    page_title="SUANET - Modelo simulacion incidentes",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        #'Report a bug': "jcastrog@movilidadbogota.gov.co",
        'About': "Este es un modulo integrado en el aplicativo suanet que busca correr modelos de simulaciónde incidentes de transito para recomendar distribución de unidades en campo"
    }
)


################################# PAGINA DE INICIO

st.markdown(
    """
    # BIENVENIDO AL MODULO DE SIMULACIÓN PARA PLANEACIÓN DE INCIDENTES
    """
    # En este modulo se busca simular las condiciones de atención de incidentes en campo, lo que permite ayudar a planear mejor la operación en campo de las unidades a partir de los historicos de información disponibles. El objetivo de este se centra en recomendar cantidad de recursos en campo requeridos para las metas planteadas, distribuir mejor las unidades disponibles en campo para atender de manera eficiente las áreas con mayor probabilidad de eventos, y demás funciones requeridas por el equipo para optimizar la atención.
    
    # Para esto, se distribuyen las unidades disponibles y se simula **hora por hora** como se atienden los incidnetes historicos de determinada fecha.
)
### SESSON STATES
if 'expander_params' not in st.session_state:
    st.session_state.expander_params = True
if 'menu_option' not in st.session_state:
    st.session_state.menu_option = 0
if 'enviar' not in st.session_state:
    st.session_state.enviar = False
    
def reiniciar():
    st.session_state.count_sends1 = st.session_state.count_sends1+1
# def contract():
#     #st.session_state.expander_params = not st.session_state.expander_params
#     st.session_state.expander_params = False
    
def contract():
    if st.session_state.expander_params:
        st.session_state.expander_params = False
    else:
        st.session_state.expander_params = True

def send():
    st.session_state.enviar = True
    contract()
    # st.experimental_singleton.clear()
    st.cache_resource.clear()
    # #ESTO BORRA
    # st.session_state['menu_option'] = "Simulación"
    # if st.session_state.get('menu_4', False):
    #     st.session_state['menu_4'] = 0
    ## ESTO CORRE
    st.session_state['menu_option'] = 0
    manual_select = st.session_state['menu_option']

with st.expander("PARÁMETROS DE SIMULACIÓN", expanded=st.session_state.expander_params):
    with st.form("my_form"):
        st.markdown("#### DEFINIR PARÁMETROS DE SIMULACIÓN")
        st.markdown('''
                    - Selecciona las fechas historicas de simulación (no mas de 5 dias por ahora)
                    - La ubicación de las unidades es aleatoria por ahora, se planea tener la opción de ubicarlos con historicos
                    - Los desplazamientos pueden ser lineales o a traves de malla de tiempo de respuesta''')
        header1 = st.columns([1,1])
        row1 = st.columns([1,1])
        fecha_inicio = row1[0].date_input("Define la fecha de inicio de la simulación", min_value=dt.datetime(2024,1,1), value=datetime.today() - timedelta(days=8))#,max_value=dt.datetime(2022,12,31))
        fecha_fin = row1[1].date_input("Define la fecha de fin de la simulación", min_value=dt.datetime(2024,1,1), value=datetime.today() - timedelta(days=1))#, max_value=dt.datetime(2022,12,31))
        row4 = st.columns([1,1])
        dist_unidades = row4[0].selectbox("Distribución de las unidades", options=("Uniforme","Optimo"))
        dias = row4[1].multiselect("Selecciona los dias de la semana entre las fechas elegidas para simular incidentes",['Lunes','Martes','Miercoles','Jueves','Viernes','Sabado','Domingo','Todos los dias'], default='Todos los dias', help='Este campo filtra los dias entre las fehcas de inicio y fin seleccionadas arriba. Se pueden seleccionar una o varias',placeholder='Seleccione una o varias opciones')
        row2 = st.columns([1,1])
        ubicacion_unidades = row2[0].selectbox("Ubicacion de unidades", options=("Aleatoria","Centroides"))
        modo_desplazamiento = row2[1].selectbox("Modo de desplazamiento de las unidades al evento", options=("Lineal","Malla"))
        
        header2 = st.columns([1,1,1])
        row3 = st.columns([1,1,1])
        policias = row3[0].slider("Cantidad de policias para simulación",0,1000,100)
        agentes = row3[1].slider("Cantidad de agentes para simulación",0,1000,10)
        gogev = row3[2].slider("Cantidad de guias para simulación",0,1000,10)
        
        # submitted = st.form_submit_button('CORRER SIMULACIÓN')
        st.form_submit_button('CORRER SIMULACIÓN', on_click=send)
    col1, col2, col3 = st.columns([1.5,1,1])
    col2.button(f"BORRAR SIMULACIÓN", key='switch_button')
# 4. Manual Item Selection
if st.session_state.get('switch_button', False):
    # ## ESTO CORRE
    # st.session_state['menu_option'] = 0
    # manual_select = st.session_state['menu_option']
    #ESTO BORRA
    st.session_state['menu_option'] = "Simulación"
    if st.session_state.get('menu_4', False):
        st.session_state['menu_4'] = 0
else:
    manual_select = None

def streamlit_menu():
    selected = option_menu(
                menu_title=None,  # required
                options=["Simulación", "Distribución de unidades simulada","Resultados"],  # required
                menu_icon="cast",  # optional
                default_index=0,  # optional
                orientation="horizontal",
                manual_select=st.session_state['menu_option'], 
                key='menu_4'
            )
    return selected
selected = streamlit_menu()
# st.session_state.menu_select = streamlit_menu()
    
@st.cache_data
def consult_data(fecha_inicio0, fecha_fin0):
    fecha_fin0 = fecha_fin0+timedelta(days=1)
    fecha_inicio = fecha_inicio0.strftime('%Y-%m-%d %H:%M:%S')
    fecha_fin = fecha_fin0.strftime('%Y-%m-%d %H:%M:%S')
    ## CONEXIÓN A LA BASE DE DATOS
    dialect = 'oracle'
    sql_driver = 'oracledb'
    ## ORACLE SDM ## hacer esto con variables de entorno
    un = os.environ["UNSN"]
    host = os.environ["HOST"]
    port = os.environ["PORT"]
    sn = os.environ["UNSN"]
    pw = os.environ["P"]
    try:
        if (fecha_fin0-fecha_inicio0).days <8:
            to_engine: str = dialect + '+' + sql_driver + '://' + un + ':' + pw + '@' + host + ':' + str(port) + '/?service_name=' + sn
            connection = sa.create_engine(to_engine)
            query = f"SELECT INCIDENTNUMBER, LATITUDE, LONGITUDE, INCIDENTDATE FROM MV_INCIDENT WHERE INCIDENTDATE BETWEEN TO_TIMESTAMP('{fecha_inicio}', 'YYYY-MM-DD HH24:MI:SS') AND TO_TIMESTAMP('{fecha_fin}', 'YYYY-MM-DD HH24:MI:SS')"
            test_df = pd.read_sql_query(query, connection)
            test_df.columns = ['INCIDENTNUMBER','LATITUDE','LONGITUDE','INCIDENT_TIME']
            test_df = test_df[['LATITUDE','LONGITUDE','INCIDENT_TIME']]
            test_df = test_df[test_df['LATITUDE']!=0]
            
            st.info(str(test_df.shape[0]) + " incidentes consultados desde base de datos entre " + str(test_df.INCIDENT_TIME.min()) + " y " + str(test_df.INCIDENT_TIME.max()))
            return test_df
        else:
            st.error("Por favor seleccione una cantidad de dias mayor a 1 y menor a 8.")
            return pd.DataFrame(columns = ['LATITUDE','LONGITUDE','INCIDENT_TIME'])
    except:
            data = pd.read_feather(os.path.join("data", "incidentes_sdm_2022_2023"))
            data = data[['LATITUDE','LONGITUDE','INCIDENT_TIME']]
            data['INCIDENT_TIME'] = pd.to_datetime(data['INCIDENT_TIME'])
            st.success('CONSULTADOS DATOS FIJOS')
            return data

# if submitted:
if st.session_state.enviar:
    contract()
    ###################### DATA
    #data = gpd.read_file("../../TRATAMIENTO DATA/outs/incidentes_sdm_2019_2023.geojson", encoding='latin-1')
    with st.spinner('Ejecutando...'):
        ## Importar datos - Futura conexión DBOracle
        #data = pd.read_feather(r"data\incidentes_sdm_2022_2023")
        consulta = st.empty()
        data = consult_data(fecha_inicio, fecha_fin)
        ## Esta es fija y requiere data de barrios en la carpeta
        # barrios = gpd.read_file(r"data\barrios-bogota.zip")
        barrios = gpd.read_file(os.path.join("data","barrios-bogota.zip"))
        barrios = barrios.to_crs(epsg=4326)
        quitar = [301,331,695,763, 762, 715,710, 1073, 829, 1055, 830, 1145,744,322,372,394, 1067,56,710, 1085, 731, 1070,1102,1106, 1104, 260, 738, 1101, 114,684, 64,40,955,413,361,1149,814,359, 422,711,705,267,310,1110,678,680,1081,161,1095,691,68,689,1110,678,578,15,313,870,745,248,989,658,284,849,497,42,639,928,551,837,434,940,550,9,283,143,535,379,526,136,945 ]
        barrios = barrios[~barrios.index.isin(quitar)]
        bogota_shape = gpd.GeoDataFrame(geometry=[cascaded_union(barrios['geometry'])], crs=barrios.crs)
        
        ## Areas policia
        # areas_pol = gpd.read_file(r"data/shp_areas_pol.shp.zip")
        areas_pol = gpd.read_file(os.path.join("data","shp_areas_pol.shp.zip"))
        bogota_shape = bogota_shape.overlay(areas_pol, how='intersection',keep_geom_type=False)
        
        ### INICIO A CORRER EL MODELO
        parametros = {
        'cant_policia' : policias,
        'cant_agentes' :agentes,
        'cant_gogev' : gogev,
        'ubicacion_unidades': ubicacion_unidades, ## Aleatoria, Centroides
        'distribucion_unidades': dist_unidades,## Uniforme, Optimo
        'metodo_desplazamiento': modo_desplazamiento, ## Lineal o malla
        'fecha_inicio': dt.datetime.combine(fecha_inicio, dt.time()),
        'fecha_fin': dt.datetime.combine(fecha_fin, dt.time()),
        'days':dias, #'Todos los dias'
        'shapefield':bogota_shape
        }
        
        if 'Todos los dias' in parametros['days']:
            incidentes = data[(data['INCIDENT_TIME']>=parametros['fecha_inicio']) & (data['INCIDENT_TIME']<=parametros['fecha_fin'])]
        else:
            trans_days = []
            for d in parametros['days']:
                if d == 'Lunes':
                    trans_days.append('Monday')
                elif d == 'Martes':
                    trans_days.append('Tuesday')
                elif d == 'Miercoles':
                    trans_days.append('Wednesday')
                elif d == 'Jueves':
                    trans_days.append('Thursday')
                elif d == 'Viernes':
                    trans_days.append('Friday')
                elif d == 'Sabado':
                    trans_days.append('Saturday')
                elif d == 'Domingo':
                    trans_days.append('Sunday')
            incidentes = data[(data['INCIDENT_TIME']>=parametros['fecha_inicio']) & (data['INCIDENT_TIME']<=parametros['fecha_fin'])]
            incidentes = incidentes[incidentes['INCIDENT_TIME'].dt.day_name().isin(trans_days)]
        #st.dataframe(incidentes)

        if incidentes.shape[0] == 0:
            st.write("Se requiere una cantidad mínima de un día para el análisis. Por favor seleccione más dias")
        elif incidentes.groupby(incidentes['INCIDENT_TIME'].dt.date).agg({'LATITUDE':'count'}).reset_index().shape[0]>=10:
            st.write("Selecciono demasiados dias. Por favor seleccione menos dias")
        else:
            global sim1
            if selected == "Simulación":
                contract()
                # sim1 = sim_model.incident_plan_simulation(incidentes, parametros)
                # #Expander simulacion  
                sim1 = sim_model.incident_plan_simulation(incidentes, parametros)
                sim1.correr(graficar=True)
                st.session_state['sim'] = sim1
                st.success("Simulacion finalizada!")
            if selected == "Distribución de unidades simulada":
                contract()
                sim1 = st.session_state['sim']
                st.markdown("## 1. Mapa de distribución de unidades con ubicación -"+str(parametros['ubicacion_unidades'])+"-, y distribución -"+str(parametros['distribucion_unidades']+"- y el centroide de los eventos en las fechas ingresadas:"))
                cols = st.columns([2,1])
                mapa, tabla = sim1.distribuir_unidades(graficar_dist=True)
                cols[0].plotly_chart(mapa, use_container_width=True)
                cols[1].write(tabla)
            if selected == "Resultados":
                contract()
                #sim1 = crear_simulacion(incidentes, parametros)
                #sim1.correr(graficar=False)
                # Expander resultados
                sim1 = st.session_state['sim']
                st.markdown("# Reporte de simulación")
                st.markdown("## 2. Reporte de tiempos de atención simulados")
                st.write(sim1.reportar())