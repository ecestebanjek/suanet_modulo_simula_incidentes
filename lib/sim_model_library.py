import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.ops import cascaded_union
from shapely.geometry import Point
import numpy as np
from datetime import datetime, timedelta
import math
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import random
import folium
import time
import plotly.express as px
import plotly.figure_factory as ff

from streamlit_folium import st_folium, folium_static
import streamlit as st

class incidente():
    """ 
    Define la clase incidente, sus tipos, y estados.
    Atributos
    Estado: Atendido, No atendido, No ocurrido
    Tipo: Choque simple, Chique con herido...
    Tiempo_respuesta: NA, integer en minutos del tiempo de respuesta
    
    Metodos:
    Ocurrir: Cambia el estado de No ocurrido a no atendido
    Atender: Cambia el estado de No Antendido a Atendido
    """
    def __init__(self, latitud, longitud, timest):
        self.estado = "No ocurrido"
        self.tipo = "Choque simple"
        self.tiempo_respuesta = np.nan
        self.latitud = latitud
        self.longitud = longitud
        self.time = timest
        self.unidad_asignada = ""
        self.t_rta = np.nan
        self.h_ocurre = 0
        self.h_atend = 0
        
    def ocurrir(self):
        self.estado = "No atendido"
    
    def atender(self):
        self.estado = "Atendido"
        
class unidad():
    """
    Define la clase unidad, sus tipos y estados
    Atributos:
    - Unidad: GOGEV, Agente, Policia
    - Estados: Atendiendo, Libre
    
    Metodos:
    - Ocupar: Cambia el estado de Libre a Atendiendo
    - Liberar: Cambia el estado de Atendiendo a Libre
    """
    def __init__(self, unidad, lat, lon):
        self.unidad = unidad
        self.estado = "Libre"
        self.prox_hora = np.nan
        self.latitud = lat#4.647962
        self.longitud = lon#-74.10328
        self.horas_para_liberar = 0
    
    def ocupar(self):
        self.estado = "Ocupado"
    def liberar(self):
        self.estado = "Libre"


class incident_plan_simulation:
    ### DESCRIPCIÓN
    """Corre una simulacion instanciando los incidentes y unidades en cada dia, y corriendo por el tiempo y cambiando los estados de los incidentes y las unidades. Para cada incidente calcula un tiempo de respuesta.
    Atributos:
    ubicacion_unidades:
        - Aleatoria
        - Historica
        - Propuesta
    metodo_desplazamiento:
        - Linea recta
        - A traves de una matriz de tiempos de viaje
    fecha_inicio:
    fecha_fin:
    lista_incidentes:
    cant_unidades_policia:
    cant_unidades_agentes:
    cant_unidades_gogev:
    resultados:

    Metodos:
    correr: Va hora por hora evolucionando estados
    reportar: Genera el reporte general de la simulación, es decir, un listado con los tiempos de respuesta, y el resumen de estadisticas descriptivas y resumen de eventos atendidos
    """
    
    ## CONSTRUCTOR DE LA CLASE
    def __init__(self, incidentes,parametros):
        cant_policia = parametros['cant_policia']
        cant_agentes = parametros['cant_agentes']
        cant_gogev = parametros['cant_gogev']
        ubic_unidades = parametros['ubicacion_unidades']
        metodo_desp = parametros['metodo_desplazamiento']
        fecha_ini = parametros['fecha_inicio']
        fecha_f = parametros['fecha_fin']
        shape = parametros['shapefield']
        d_unid = parametros['distribucion_unidades']
        
        self.shapefield = shape
        self.cant_unidades_policia = cant_policia
        self.cant_unidades_agentes = cant_agentes
        self.cant_unidades_gogev = cant_gogev
        self.ubicacion_unidades = ubic_unidades
        self.metodo_desplazamiento = metodo_desp
        self.modo_distribucion_unidades = d_unid
        self.fecha_inicio = fecha_ini
        self.fecha_fin = fecha_f
        self.incidentes = incidentes # Dataframe con ubicacion y timestamp de incidentes
        self.policias = {}
        self.agentes = {}
        self.gogev = {}
        self.eventos_por_atender = {}
        self.eventos_atendidos = {}
        self.zat = gpd.read_file(r"data\ZAT\zat.shp")
        self.matriz =pd.read_feather(r"data\origen_destino")
        self.dist_unidades = list()
        
    
    ###  DISTRIBUCION DE UNIDADES EN AREAS - Retorna la cantidad de agentes por area
    def distribuir_unidades(self,graficar_dist=False):
        ### Distribución dependiendo de la opcion seleccionada
        if self.modo_distribucion_unidades == "Uniforme":
            # Dist policias:
            #suma = np.sum([self.cant_unidades_policia, self.cant_unidades_agentes,self.cant_unidades_gogev ])
            suma_p = self.cant_unidades_policia
            suma_a = self.cant_unidades_agentes
            suma_g = self.cant_unidades_gogev
            num_p = suma_p//self.shapefield.shape[0]
            num_a = suma_a//self.shapefield.shape[0]
            num_g = suma_g//self.shapefield.shape[0]
            lista_p = [num_p] * self.shapefield.shape[0]  # Crea una lista de elementos iguales a 'num'
            lista_a = [num_a] * self.shapefield.shape[0]  # Crea una lista de elementos iguales a 'num'
            lista_g = [num_g] * self.shapefield.shape[0]  # Crea una lista de elementos iguales a 'num'
            suma_actual_p = sum(lista_p)  # Calcula la suma actual de la lista
            suma_actual_a = sum(lista_a)  # Calcula la suma actual de la lista
            suma_actual_g = sum(lista_g)  # Calcula la suma actual de la lista
            n = 7#posicion a la cual aumentar
            while suma_actual_p != suma_p:
                diferencia = suma_actual_p - suma_p
                reducir = diferencia // self.shapefield.shape[0] # Calcula el valor a reducir en cada elemento de la lista
                lista_p[n] -= reducir  # Reduce el primer elemento de la lista
                n -=1
                suma_actual_p = sum(lista_p)  # Calcula la nueva suma actual
            n = 7#posicion a la cual aumentar
            while suma_actual_a != suma_a:
                diferencia = suma_actual_a - suma_a
                reducir = diferencia // self.shapefield.shape[0] # Calcula el valor a reducir en cada elemento de la lista
                lista_a[n] -= reducir  # Reduce el primer elemento de la lista
                n -=1
                suma_actual_a = sum(lista_a)  # Calcula la nueva suma actual
            n = 0#posicion a la cual aumentar
            while suma_actual_g != suma_g:
                diferencia = suma_actual_g - suma_g
                reducir = diferencia // self.shapefield.shape[0] # Calcula el valor a reducir en cada elemento de la lista
                lista_g[n] -= reducir  # Reduce el primer elemento de la lista
                n -=1
                suma_actual_g = sum(lista_g)  # Calcula la nueva suma actual
            areas = ['AREA 1','AREA 2','AREA 3','AREA 4','AREA 5','AREA 6','AREA 7','AREA 8','AREA 9','AREA 10','AREA 11','AREA 12','AREA 13','AREA 14','AREA 15']
            self.dist_unidades = pd.DataFrame({'areas':areas, 'dist_p':lista_p, 'dist_a':lista_a, 'dist_g':lista_g})
        elif self.modo_distribucion_unidades == "Optimo":
            inc = gpd.GeoDataFrame(self.incidentes,geometry=gpd.points_from_xy(self.incidentes.LONGITUDE, self.incidentes.LATITUDE), crs="EPSG:4326")
            inc_area =  self.shapefield.sjoin(inc, how='left', predicate='intersects').groupby('area_polic').agg({'index_right':'count'}).reset_index()
            inc_area['pond'] = inc_area['index_right']/np.sum(inc_area['index_right'])
            lista_p = np.ceil(self.cant_unidades_policia*inc_area['pond']).astype(int)
            lista_a = np.ceil(self.cant_unidades_agentes*inc_area['pond']).astype(int)
            lista_g = np.ceil(self.cant_unidades_gogev*inc_area['pond']).astype(int)
            suma_actual_p = sum(lista_p)  # Calcula la suma actual de la lista
            suma_actual_a = sum(lista_a)  # Calcula la suma actual de la lista
            suma_actual_g = sum(lista_g)  # Calcula la suma actual de la lista
            suma_p = self.cant_unidades_policia
            suma_a = self.cant_unidades_agentes
            suma_g = self.cant_unidades_gogev
            n = 0#posicion a la cual aumentar
            while suma_actual_p != suma_p:
                lista_p[n] -= 1  # Reduce el primer elemento de la lista
                n +=1
                if n==14:
                    n=0
                suma_actual_p = sum(lista_p)  # Calcula la nueva suma actual
            n = 0#posicion a la cual aumentar
            while suma_actual_a != suma_a:
                lista_a[n] -= 1  # Reduce el primer elemento de la lista
                n +=1
                if n==14:
                    n=0
                suma_actual_a = sum(lista_a)  # Calcula la nueva suma actual
            n = 0#posicion a la cual aumentar
            while suma_actual_g != suma_g:
                lista_g[n] -= 1  # Reduce el primer elemento de la lista
                n +=1
                if n==14:
                    n=0
                suma_actual_g = sum(lista_g)  # Calcula la nueva suma actual
            inc_area['dist_p'] = lista_p
            inc_area['dist_a'] = lista_a
            inc_area['dist_g'] = lista_g
            inc_area['areas'] = inc_area['area_polic']
            self.dist_unidades = inc_area[['areas','dist_p','dist_a','dist_g']]
        if graficar_dist:
            geo_df = self.shapefield.merge(self.dist_unidades, how='left', left_on='area_polic', right_on='areas')
            geo_df['Distribucion'] = geo_df['dist_p']+geo_df['dist_a']+geo_df['dist_g']
            inc = gpd.GeoDataFrame(self.incidentes,geometry=gpd.points_from_xy(self.incidentes.LONGITUDE, self.incidentes.LATITUDE), crs="EPSG:4326")
            inc_area =  inc.to_crs("epsg:3857").sjoin(self.shapefield.to_crs("epsg:3857"), how='left', predicate='intersects').dissolve(by='area_polic', aggfunc='mean').reset_index()
            lons = inc_area.geometry.centroid.to_crs(crs=4326).x
            lats = inc_area.geometry.centroid.to_crs(crs=4326).y
            fig = px.choropleth_mapbox(geo_df, height=800, width=800,
                           geojson=geo_df.geometry,
                           locations=geo_df.index,
                           color="Distribucion",hover_name =geo_df.areas,
                           center={"lat": 4.65, "lon": -74.1},
                           opacity=0.5,color_continuous_scale="Viridis",
                           mapbox_style="open-street-map",
                           zoom=10)
            fig.add_scattermapbox(
                lat = lats,
                lon = lons,
                mode = 'markers+text',
                text = 'cantroide',
                marker_size=12,
                marker_color='rgb(235, 0, 100)'
                )
            #fig.show()
            #print(self.dist_unidades)
            return fig, self.dist_unidades.rename(columns={"dist_p": "CANTIDAD POLICIA",
                                                           "dist_a": "CANTIDAD AGENTES",
                                                           'dist_g': "CANTIDAD GOGEV"})

    # def dist_para_optimizar(self, arr):## Recibe un arreglo de 3x15 con distribucion de agentes, policias y gogev
    #     areas = ['AREA 1','AREA 2','AREA 3','AREA 4','AREA 5','AREA 6','AREA 7','AREA 8','AREA 9','AREA 10','AREA 11','AREA 12','AREA 13','AREA 14','AREA 15']
    #     lista_p = arr[0]
    #     lista_a = arr[1]
    #     lista_g = arr[2]
    #     self.dist_unidades = pd.DataFrame({'areas':areas, 'dist_p':lista_p, 'dist_a':lista_a, 'dist_g':lista_g})

    ### CREA LAS INSTANCIAS DE LAS UNIDADES
    def crear_instancias_unid(self):
        cont_pol = 0
        cont_ag = 0
        cont_gog = 0
        for i in range(len(self.shapefield)):
            #Policias
            cant_pol = self.dist_unidades[self.dist_unidades['areas']==self.shapefield['area_polic'][i]].reset_index()['dist_p'][0]
            coord = self.generar_coordenadas_aleatorias(self.shapefield.iloc[i:i+1] ,cant_pol)
            for j in range(cant_pol):
                self.policias[f"pol_{j+cont_pol}"] = unidad("Policia", coord[j][1], coord[j][0])
            cont_pol += cant_pol
            #Agentes
            cant_ag = self.dist_unidades[self.dist_unidades['areas']==self.shapefield['area_polic'][i]].reset_index()['dist_a'][0]
            coord = self.generar_coordenadas_aleatorias(self.shapefield.iloc[i:i+1] ,cant_ag)
            for j in range(cant_ag):
                self.agentes[f"age_{j+cont_ag}"] = unidad("Agente", coord[j][1], coord[j][0])
            cont_ag += cant_ag
            #Gogev
            cant_gog = self.dist_unidades[self.dist_unidades['areas']==self.shapefield['area_polic'][i]].reset_index()['dist_g'][0]
            coord = self.generar_coordenadas_aleatorias(self.shapefield.iloc[i:i+1] ,cant_gog)
            for j in range(cant_gog):
                self.gogev[f"gog_{j+cont_ag}"] = unidad("Gogev", coord[j][1], coord[j][0])
            cont_gog += cant_gog
    
    #### FUNCIONES DE CALCULO DE DISTANCIA LINEAL
    def calcular_distancia(self,coordenadas1, coordenadas2):
        lat1, lon1 = coordenadas1
        lat2, lon2 = coordenadas2

        # Conversión de grados a radianes
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Fórmula de la distancia entre dos puntos en la Tierra utilizando la fórmula de Haversine
        distancia = 2 * math.asin(math.sqrt(math.sin((lat2_rad - lat1_rad) / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin((lon2_rad - lon1_rad) / 2) ** 2))
        # Radio de la Tierra en kilómetros
        radio_tierra = 6371
        # Distancia en kilómetros
        distancia *= radio_tierra

        return distancia
    
    #### FUNCION QUE ENCUENTRA EL OBJETO MAS CERCANO LINEALMENTE
    def encontrar_objeto_mas_cercano(self,evento_coordenadas, diccionario_objetos,t):
        objeto_mas_cercano = None
        distancia_minima = float('inf')

        for key, objeto_unidad in diccionario_objetos.items():
            objeto_coordenadas = (objeto_unidad.latitud, objeto_unidad.longitud)
            distancia = self.calcular_distancia(evento_coordenadas, objeto_coordenadas)
            if distancia < distancia_minima:
                distancia_minima = distancia
                objeto_mas_cercano = key
        #print(distancia_minima, end='\r')
        a=distancia_minima/30*60
        t.write("Tiempo de viaje calculado al evento: "+str(a))
        return objeto_mas_cercano, distancia_minima
    
    #### FUNCION QUE ENCUENTRA EL OBJETO MAS CERCANO SOBRE MALLA VIAL
    def buscar_objeto_mas_cercano_malla_vial(self, evento_coordenadas, diccionario_objetos,t):
        """Las coordenadas vienen (lat,lon),
           Cada objeto del diccionario tiene latitud y longitud"""
        ## En que ZAT esta el evento
        evento = pd.DataFrame([evento_coordenadas], columns=['lat','lon'])
        evento = gpd.GeoDataFrame(evento, geometry=gpd.points_from_xy(evento.lon, evento.lat), crs="EPSG:4686")
        evento = evento.sjoin(self.zat, how='left', predicate='within')
        evento.reset_index(inplace=True, drop=True)
        ## En que ZAT esta cada unidad libre
        coords_lat = []
        coords_lon = []
        k = []
        
        for key, objeto_unidad in diccionario_objetos.items():
            coords_lat.append(objeto_unidad.latitud)
            coords_lon.append(objeto_unidad.longitud)
            k.append(key)
        ubicacion_objetos = pd.DataFrame([k, coords_lat,coords_lon]).T
        ubicacion_objetos.columns = ['key','lat','lon']
        ubicacion_objetos = gpd.GeoDataFrame(ubicacion_objetos, geometry=gpd.points_from_xy(ubicacion_objetos.lon, ubicacion_objetos.lat), crs="EPSG:4686")
        ubicacion_objetos = ubicacion_objetos.sjoin(self.zat, how='left', predicate='within')
        ## Se hacen los pares de ZATs
        ubicacion_objetos['ZAT_y'] = evento['ZAT'].iloc[0]
        ## Calcula distancias
        ubicacion_objetos = ubicacion_objetos.merge(self.matriz, how='left', left_on=['ZAT', 'ZAT_y'], right_on=['x','y'])
        min_time = np.min(ubicacion_objetos['time'])
        #print(min_time, end='\r')
        t.write("Tiempo de viaje calculado al evento:" + str(min_time))
        min_obj_key = ubicacion_objetos[ubicacion_objetos['time'] == min_time].reset_index()
        try:    
            min_obj_key = min_obj_key['key'].iloc[0]
        except:
            print("Objeto no encontrado")
            print(min_obj_key)
            min_obj_key = 8 #Promedio
        
        return min_obj_key, min_time

    #### FUNCION DE GENERACION ALEATORIA DE COORDENADAS
    def generar_coordenadas_aleatorias(self, geodataframe, cantidad):
        geodataframe.reset_index(inplace=True, drop=True)
        area = geodataframe['area_polic'][0]
        poligono = geodataframe.geometry[0]#.iloc[0]  # Obtener el polígono del GeoDataFrame
        ## centroide areas
        inc = gpd.GeoDataFrame(self.incidentes,geometry=gpd.points_from_xy(self.incidentes.LONGITUDE, self.incidentes.LATITUDE), crs="EPSG:4326")
        inc_area =  inc.to_crs("epsg:3857").sjoin(self.shapefield.to_crs("epsg:3857"), how='left', predicate='intersects').dissolve(by='area_polic').reset_index()
        #inc_area =  gpd.GeoDataFrame(inc_area[inc_area['area_polic'] == area]).reset_index()
        inc_area =  inc_area[inc_area['area_polic'] == area].reset_index()
        centroide_x = inc_area[0:1].geometry.centroid.to_crs(crs=4326).x
        centroide_y = inc_area[0:1].geometry.centroid.to_crs(crs=4326).y
        minx, miny, maxx, maxy = poligono.bounds  # Obtener los límites del polígono
        coordenadas = []
        if self.ubicacion_unidades == "Aleatoria":
            while len(coordenadas) < cantidad:
                x = np.random.uniform(minx, maxx)
                y = np.random.uniform(miny, maxy)
                punto = Point(x, y)

                if poligono.contains(punto):
                    coordenadas.append((x, y))
        
        if self.ubicacion_unidades == "Centroides":
            while len(coordenadas) < cantidad:
                x = np.random.normal(centroide_x, (maxx-minx)/12)
                y = np.random.normal(centroide_y, (maxy-miny)/12)
                punto = Point(x, y)

                if poligono.contains(punto):
                    coordenadas.append((x, y))

        return coordenadas

    #### FUNCION QUE CORRE EL MODELO DE SIMULACIÓN DE ATENCIÓN
    def correr(self,graficar=False):
        #m = folium.Map(location=[4.60971, -74.08175], tiles="OpenStreetMap", zoom_start=11)
        t = st.empty()
        map = st.empty()
        # Inicio de variables de simulación
        intervalo = timedelta(hours=1)
        fecha_actual = self.fecha_inicio
        global anterior
        anterior = self.fecha_inicio
        # Crea las instancias de unidades opciones: Aleatoria, aleatoria cerca a centroides, Por pesos de incidentes, 
        self.distribuir_unidades()
        self.crear_instancias_unid()
        contador_incidentes = 0
        while fecha_actual <= self.fecha_fin:
            """ ACA VA LA LOGICA DE SIMULACION CADA DIA"""
            """
            1- Crear los objetos de los incidentes de esa hora
            2- Asignar una unidad disponible a cada evento en el listado de incidentes sin atender
                a. Para asignar una unidad, debe calcular el tiempo de viaje de todas las unidades a cada evento, y seleccionar la unidad de menor tiempo
                b. Asignar la unidad - Ocupar la unidad por 2 horas
                c. Guardar en el incidnete atendido el tiempo de respuesta
                
            """
            
            ### Distribucion aleatoria de agentes cada dia:
            if anterior!=fecha_actual.date():
                self.crear_instancias_unid()

            ### Le quita 1 al contador de tiempo para liberar, y si es =0 libera la unidad
            for (id, obj) in self.policias.items():
                if obj.horas_para_liberar >0:
                    self.policias[id].horas_para_liberar -=1
                if obj.horas_para_liberar ==0:
                    self.policias[id].liberar()
            
            for (id, obj) in self.agentes.items():
                if obj.horas_para_liberar >0:
                    self.agentes[id].horas_para_liberar -=1
                if obj.horas_para_liberar ==0:
                    self.agentes[id].liberar()
            
            for (id, obj) in self.gogev.items():
                if obj.horas_para_liberar >0:
                    self.gogev[id].horas_para_liberar -=1
                if obj.horas_para_liberar ==0:
                    self.gogev[id].liberar()
            #######
            
            incidentes_dia_hora = self.incidentes[(self.incidentes['INCIDENT_TIME'].dt.date == fecha_actual.date()) & (self.incidentes['INCIDENT_TIME'].dt.hour == fecha_actual.hour)]
            if len(incidentes_dia_hora)>0:
                for i in range(len(incidentes_dia_hora)):
                    contador_incidentes += 1
                    self.eventos_por_atender[f"ev_{contador_incidentes}"] = incidente(incidentes_dia_hora.iloc[i]['LATITUDE'], incidentes_dia_hora.iloc[i]['LONGITUDE'], incidentes_dia_hora.iloc[i]['INCIDENT_TIME'])
                    self.eventos_por_atender[f"ev_{contador_incidentes}"].estado = "No atendido"
                    self.eventos_por_atender[f"ev_{contador_incidentes}"].h_ocurre = fecha_actual


            ## Calcular tiempos para cada evento por atender
            for id, obj_inc in self.eventos_por_atender.copy().items():
                ## Crea listas de unidades NO OCUPADAS
                unid_libres = dict()
                for (key, value) in self.policias.items():
                    # Check if key is even then add pair to new dictionary
                    if value.estado == "Libre":
                        unid_libres[key] = value
                for (key, value) in self.agentes.items():
                    # Check if key is even then add pair to new dictionary
                    if value.estado == "Libre":
                        unid_libres[key] = value
                for (key, value) in self.gogev.items():
                    # Check if key is even then add pair to new dictionary
                    if value.estado == "Libre":
                        unid_libres[key] = value
                ###
                ####### SI NO HAY UNIDADES DISPONIBLES, NO HAGA NADA
                if len(unid_libres)==0:
                    print("Hora con no disponibles")
                else:  
                    # Funcion que asigna un id de unidad para cada evento segun distancia
                    x_ev = obj_inc.latitud
                    y_ev = obj_inc.longitud
                    evento_coordenadas = (x_ev, y_ev)
                    ### funcion que recibe coordenadas de incidentes y lista de unidades libres y retorne el id de la unidad mas cercana y su tiempo de viaje
                    if self.metodo_desplazamiento=="Lineal":
                        objeto_mas_cercano, distancia_al_evento = self.encontrar_objeto_mas_cercano(evento_coordenadas, unid_libres,t)
                        tiempo_al_evento = distancia_al_evento/30*60 # Se asume velocidad de 30 km/h de cada unidad, se pasa a minutos
                    elif self.metodo_desplazamiento=="Malla":
                        objeto_mas_cercano, tiempo_al_evento = self.buscar_objeto_mas_cercano_malla_vial(evento_coordenadas, unid_libres,t)
                    ## Asignar unidad al evento (Crear la variable dentro de la clase)
                    self.eventos_por_atender[id].unidad_asignada = objeto_mas_cercano
                    ## Asignarle al evento el tiempo de respuesta
                    self.eventos_por_atender[id].t_rta = tiempo_al_evento
                    self.eventos_por_atender[id].h_atend = fecha_actual
                    delta = self.eventos_por_atender[id].h_atend - self.eventos_por_atender[id].h_ocurre
                    delta = delta.total_seconds()/60
                    self.eventos_por_atender[id].t_rta = self.eventos_por_atender[id].t_rta+delta
                    ## copia el evento a la lista de eventos atendidos
                    self.eventos_atendidos[id] = self.eventos_por_atender[id]
                    ## Elimina el evento de la lista de eventos por atender
                    self.eventos_por_atender.pop(id)
                    ## Cambia el estado de la unidad a ocupado, y le asigna un valor al contador de horas ocuapdas entero(t_rta + t_atendiento)
                    if objeto_mas_cercano in self.policias:
                        self.policias[objeto_mas_cercano].ocupar()
                        self.policias[objeto_mas_cercano].horas_para_liberar = math.ceil((120+tiempo_al_evento)/60)
                    if objeto_mas_cercano in self.agentes:
                        self.agentes[objeto_mas_cercano].ocupar()
                        self.agentes[objeto_mas_cercano].horas_para_liberar =  math.ceil((120+tiempo_al_evento)/60)
                    if objeto_mas_cercano in self.gogev:
                        self.gogev[objeto_mas_cercano].ocupar()
                        self.gogev[objeto_mas_cercano].horas_para_liberar =  math.ceil((120+tiempo_al_evento)/60)            
            
            ## Barra de progreso del loop : https://medium.com/@harshit4084/track-your-loop-using-tqdm-7-ways-progress-bars-in-python-make-things-easier-fcbbb9233f24
            if (anterior!=fecha_actual.date()) & (graficar==True) & (fecha_actual != self.fecha_fin):
                #clear_output(wait=True)
                # m = self.plot_map_in_time(fecha_actual,m)
                # st_data = folium_static(m)
                #time.sleep(5)
                fig = self.plot_map_in_time(fecha_actual)
                map.plotly_chart(fig, use_container_width=True)
                
                
            anterior = fecha_actual.date()
            fecha_actual += intervalo
    
    #### FUNCION QUE HACE EL REPORTE, GRAFICAS DE RESULTADOS Y DEMAS
    def reportar(self):
        tiempos = []
        ids = []
        for (key, value) in self.eventos_atendidos.items():
            # Check if key is even then add pair to new dictionary
            tiempos.append(value.t_rta)
            ids.append(key)
        rep = pd.DataFrame([ids, tiempos]).T
        rep.columns = ['ids','Tiempo de viaje a atención']
        rep['Tiempo de viaje a atención'] =  pd.to_numeric(rep['Tiempo de viaje a atención'])
        rep['Tiempo de respuesta (supuesto asinación 10 mins)'] = rep['Tiempo de viaje a atención'] + 10
        # st.write(rep[['t_rta','t_rta_tot']].describe())
        rep.dropna(subset='Tiempo de viaje a atención', inplace=True)
        #plt.hist(rep.t_rta, bins = 100)
        #fig = px.histogram(rep, x="t_rta", nbins=20,labels={'x':'Tiempo de respuesta simulado', 'y':'Cantidad'},title='Histograma de tiempo de respuesta simulado',)
        fig = ff.create_distplot([rep['Tiempo de viaje a atención'], rep['Tiempo de respuesta (supuesto asinación 10 mins)']], ['Tiempos de viaje simulados','Tiempos de respuesta con supuesto de 10 mins en asignación'],bin_size=.2, show_rug=False)
        fig.update_layout(title_text='Distribución de tiempos de atención calculados')
        col = st.columns([1,2])
        col[0].write(rep[['Tiempo de viaje a atención','Tiempo de respuesta (supuesto asinación 10 mins)']].describe())
        col[1].plotly_chart(fig, use_container_width=True)
        #return figure
    
    ### FUNCION QUE HACE LOS MAPAS
    def plot_map(self):
        inc = self.incidentes[(self.incidentes['INCIDENT_TIME'].dt.date >= self.fecha_inicio.date()) & (self.incidentes['INCIDENT_TIME'].dt.date <= self.fecha_fin.date())]
        pol = []
        for (key, obj) in self.policias.items():
            pol.append((key, obj.latitud, obj.longitud))
        pol = pd.DataFrame(pol, columns=['key','lat','lon'])
        
        age = []
        for (key, obj) in self.agentes.items():
            age.append((key, obj.latitud, obj.longitud))
        age = pd.DataFrame(age, columns=['key','lat','lon'])
        
        gog = []
        for (key, obj) in self.gogev.items():
            gog.append((key, obj.latitud, obj.longitud))
        gog = pd.DataFrame(gog, columns=['key','lat','lon'])
        
        
        
        m = folium.Map(location=[4.60971, -74.08175], tiles="OpenStreetMap", zoom_start=12)
        #add marker one by one on the map
        for i in range(0,len(inc)):
            folium.CircleMarker(
                location=[inc.iloc[i]['LATITUDE'], inc.iloc[i]['LONGITUDE']],
                radius=1,
                color = 'red'
                #popup=data.iloc[i]['name'],
            ).add_to(m)
        
        for i in range(0,len(pol)):
            folium.CircleMarker(
                location=[pol.iloc[i]['lat'], pol.iloc[i]['lon']],
                radius=7,
                color = 'green',
                fill=True,
                fill_color = "green",
                fill_opacity=0.9
                #popup=data.iloc[i]['name'],
            ).add_to(m)
            
        for i in range(0,len(age)):
            folium.CircleMarker(
                location=[age.iloc[i]['lat'], age.iloc[i]['lon']],
                radius=7,
                color = 'blue',
                fill=True,
                fill_color = "blue",
                fill_opacity=0.9
                #popup=data.iloc[i]['name'],
            ).add_to(m)
        
        for i in range(0,len(gog)):
            folium.CircleMarker(
                location=[gog.iloc[i]['lat'], gog.iloc[i]['lon']],
                radius=7,
                color = 'gray',
                fill=True,
                fill_color = "gray",
                fill_opacity=0.9
                #popup=data.iloc[i]['name'],
            ).add_to(m)
        return m
    
    ### FUNCION QUE HACE LOS MAPAS
    def plot_map_in_time(self, fecha_actual):
        inc = self.incidentes[self.incidentes['INCIDENT_TIME'].dt.date == fecha_actual.date()]
        pol = []
        for (key, obj) in self.policias.items():
            pol.append((key, obj.latitud, obj.longitud))
        pol = pd.DataFrame(pol, columns=['key','lat','lon'])
        
        age = []
        for (key, obj) in self.agentes.items():
            age.append((key, obj.latitud, obj.longitud))
        age = pd.DataFrame(age, columns=['key','lat','lon'])
        
        gog = []
        for (key, obj) in self.gogev.items():
            gog.append((key, obj.latitud, obj.longitud))
        gog = pd.DataFrame(gog, columns=['key','lat','lon'])
        
        ## Con plotly
        geo_df = self.shapefield.merge(self.dist_unidades, how='left', left_on='area_polic', right_on='areas')
        inc = self.incidentes[self.incidentes['INCIDENT_TIME'].dt.date == fecha_actual.date()]
        #inc_area =  inc.to_crs("epsg:3857").sjoin(self.shapefield.to_crs("epsg:3857"), how='left', predicate='intersects').dissolve(by='area_polic', aggfunc='mean').reset_index()
        fig = px.choropleth_mapbox(geo_df, height=800, width=800,
                        geojson=geo_df.geometry,
                        locations=geo_df.index,
                        #color="Distribucion",
                        hover_name =geo_df.areas,
                        center={"lat": 4.65, "lon": -74.1},
                        opacity=0.1,color_continuous_scale="Viridis",
                        mapbox_style="open-street-map",
                        zoom=10)
        ## Añade incidentes
        fig.add_scattermapbox(
            lat = inc.LATITUDE,
            lon = inc.LONGITUDE,
            mode = 'markers+text',
            #text = 'cantroide',
            marker_size=6,
            marker_color='rgb(227, 54, 38)',
            name="Incidentes"
            )
        ## Añade policia
        fig.add_scattermapbox(
            lat = pol['lat'],
            lon = pol['lon'],
            mode = 'markers+text',
            text = 'Policia',
            marker_size=12,
            marker_color='rgb(17, 133, 19)',
            name="Policia"
            )
        ## Añade agentes
        fig.add_scattermapbox(
            lat = age['lat'],
            lon = age['lon'],
            mode = 'markers+text',
            text = 'Agente de transito',
            marker_size=12,
            marker_color='rgb(42, 51, 81)',
            name="Agentes"
            )
        ## Añade gogev
        fig.add_scattermapbox(
            lat = gog['lat'],
            lon = gog['lon'],
            mode = 'markers+text',
            text = 'GOGEV',
            marker_size=12,
            marker_color='rgb(127, 151, 92)',
            name="GOGEV"
            )
        
        #m = folium.Map(location=[4.60971, -74.08175], tiles="OpenStreetMap", zoom_start=11)
        #add marker one by one on the map
        # for i in range(0,len(inc)):
        #     folium.CircleMarker(
        #         location=[inc.iloc[i]['LATITUDE'], inc.iloc[i]['LONGITUDE']],
        #         radius=1,
        #         color = 'red'
        #         #popup=data.iloc[i]['name'],
        #     ).add_to(m)
        
        # for i in range(0,len(pol)):
        #     folium.CircleMarker(
        #         location=[pol.iloc[i]['lat'], pol.iloc[i]['lon']],
        #         radius=7,
        #         color = 'green',
        #         fill=True,
        #         fill_color = "green",
        #         fill_opacity=0.9
        #         #popup=data.iloc[i]['name'],
        #     ).add_to(m)
            
        # for i in range(0,len(age)):
        #     folium.CircleMarker(
        #         location=[age.iloc[i]['lat'], age.iloc[i]['lon']],
        #         radius=7,
        #         color = 'blue',
        #         fill=True,
        #         fill_color = "blue",
        #         fill_opacity=0.9
        #         #popup=data.iloc[i]['name'],
        #     ).add_to(m)
        
        # for i in range(0,len(gog)):
        #     folium.CircleMarker(
        #         location=[gog.iloc[i]['lat'], gog.iloc[i]['lon']],
        #         radius=7,
        #         color = 'gray',
        #         fill=True,
        #         fill_color = "gray",
        #         fill_opacity=0.9
        #         #popup=data.iloc[i]['name'],
        #     ).add_to(m)
        #st_folium(m, width=1000, height=600,key="mapa")
        return fig