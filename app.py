from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import folium
import streamlit as st
from streamlit_folium import st_folium

import routing

DEFAULT_PLACE = "San Pedro Garza García, Nuevo León, México"
FUEL_EFFICIENCY_KM_PER_L = 9.5
FUEL_PRICE_PER_L = 25.0


st.set_page_config(page_title="Rutas-A*", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 0rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

if "route_display" not in st.session_state:
    st.session_state["route_display"] = {
        "segments": None,
        "markers": None,
        "summary": None,
    }

if "garzabus_metrics" not in st.session_state:
    st.session_state["garzabus_metrics"] = None

if "last_area" not in st.session_state:
    st.session_state["last_area"] = DEFAULT_PLACE


@st.cache_resource(show_spinner=False)
def get_graph_cached(place: str):
    return routing.load_drive_graph(place)


@st.cache_data(show_spinner=False)
def geocode_cached(query: str) -> Tuple[float, float]:
    return routing.geocode(query)


def resolve_address(address: str, area: str) -> Tuple[float, float]:
    queries = [address]
    if area and area.lower() not in address.lower():
        queries.append(f"{address}, {area}")
    for query in queries:
        try:
            return geocode_cached(query)
        except Exception:
            continue
    raise ValueError(f"No se pudo geocodificar la dirección: {address}")


def format_distance(meters: float) -> str:
    kilometers = meters / 1000
    return f"{kilometers:.2f} km"


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    hours = minutes // 60
    remaining_minutes = minutes % 60
    if hours:
        return f"{hours} h {remaining_minutes} min"
    return f"{remaining_minutes} min"


def estimate_fuel_cost(distance_m: float) -> float:
    distance_km = distance_m / 1000
    liters_used = distance_km / FUEL_EFFICIENCY_KM_PER_L
    return liters_used * FUEL_PRICE_PER_L


def build_map(
    center: Tuple[float, float],
    route_segments: Optional[List[List[Tuple[float, float]]]] = None,
    markers: Optional[List[Dict[str, Any]]] = None,
) -> folium.Map:
    fmap = folium.Map(location=center, zoom_start=13, tiles="cartodbpositron")

    if route_segments:
        all_latitudes = []
        all_longitudes = []
        for segment in route_segments:
            if len(segment) < 2:
                continue
            folium.PolyLine(segment, color="#1c64f2", weight=6, opacity=0.85).add_to(fmap)
            for lat, lon in segment:
                all_latitudes.append(lat)
                all_longitudes.append(lon)
        if all_latitudes and all_longitudes:
            fmap.fit_bounds(
                [
                    (min(all_latitudes), min(all_longitudes)),
                    (max(all_latitudes), max(all_longitudes)),
                ]
            )

    if markers:
        for marker in markers:
            folium.Marker(
                location=marker["location"],
                tooltip=marker.get("tooltip"),
                popup=marker.get("popup"),
                icon=folium.Icon(color=marker.get("color", "blue"), icon="map-marker"),
            ).add_to(fmap)

    return fmap


with st.sidebar:
    st.title("Rutas-A*")
    st.write("Calcula rutas óptimas y recorridos multi-parada con A*.")
    area = st.text_input("Área de búsqueda", value=DEFAULT_PLACE)
    mode = st.selectbox(
        "Modo de Ruta",
        ("Ruta Óptima (A a B)", "Ruta del Garzabus (Multi-Parada)"),
    )

    route_summary_placeholder = st.empty()
    garzabus_metrics_container = st.container()

    if area != st.session_state["last_area"]:
        st.session_state["route_display"] = {
            "segments": None,
            "markers": None,
            "summary": None,
        }
        st.session_state["garzabus_metrics"] = None
        st.session_state["last_area"] = area

    route_segments: Optional[List[List[Tuple[float, float]]]] = st.session_state[
        "route_display"
    ]["segments"]
    markers: Optional[List[Dict[str, Any]]] = st.session_state["route_display"]["markers"]
    markers = markers or []
    graph_center = (25.6573, -100.4027)

    if area:
        try:
            graph = get_graph_cached(area)
            graph_center = graph.graph.get("center", graph_center)
            max_speed_mps = routing.compute_max_speed_mps(graph)
        except Exception as graph_error:
            st.error(f"No se pudo descargar el grafo para el área indicada. {graph_error}")
            graph = None
            max_speed_mps = 13.9
    else:
        st.error("Por favor ingresa un área válida para descargar el grafo.")
        graph = None
        max_speed_mps = 13.9

    if mode == "Ruta Óptima (A a B)":
        origin = st.text_input("Punto de Partida (A)")
        destination = st.text_input("Punto de Destino (B)")
        optimization = st.selectbox(
            "Tipo de Optimización", ["Ruta más Corta", "Ruta más Ecológica"]
        )
        if st.button("Calcular Ruta A -> B", use_container_width=True):
            if not graph:
                st.error("No se pudo calcular la ruta porque el grafo no está disponible.")
            elif not origin or not destination:
                st.error("Ingresa los puntos A y B para calcular la ruta.")
            else:
                try:
                    origin_coords = resolve_address(origin, area)
                    destination_coords = resolve_address(destination, area)
                    origin_node = routing.nearest_node(
                        graph, origin_coords[0], origin_coords[1]
                    )
                    destination_node = routing.nearest_node(
                        graph, destination_coords[0], destination_coords[1]
                    )
                    if optimization == "Ruta más Corta":
                        heuristic = routing.build_distance_heuristic(graph, destination_node)
                        result = routing.mi_astar(
                            graph, origin_node, destination_node, "length", heuristic
                        )
                        minimize_key = "length"
                    else:
                        heuristic = routing.build_time_heuristic(
                            graph, destination_node, max_speed_mps
                        )
                        result = routing.mi_astar(
                            graph,
                            origin_node,
                            destination_node,
                            "travel_time",
                            heuristic,
                        )
                        minimize_key = "travel_time"

                    metrics = routing.compute_route_metrics(
                        graph, result.nodes, minimize_key=minimize_key
                    )
                    coords = routing.route_nodes_to_coordinates(graph, result.nodes)
                    route_segments = [coords]
                    markers = [
                        {
                            "location": list(coords[0]),
                            "tooltip": "Punto de Partida",
                            "popup": origin,
                            "color": "green",
                        },
                        {
                            "location": list(coords[-1]),
                            "tooltip": "Punto de Destino",
                            "popup": destination,
                            "color": "red",
                        },
                    ]
                    summary_text = (
                        f"**Distancia:** {format_distance(metrics['length_m'])}  \
**Duración estimada:** {format_duration(metrics['travel_time_s'])}"
                    )
                    st.session_state["route_display"] = {
                        "segments": route_segments,
                        "markers": markers,
                        "summary": summary_text,
                    }
                    st.session_state["garzabus_metrics"] = None
                except Exception as calc_error:
                    st.error(f"Error al calcular la ruta: {calc_error}")

    else:
        stops_text = st.text_area(
            "Paradas (una por línea)",
            placeholder="Ejemplo:\nParada 1\nParada 2\nParada 3",
        )
        return_to_start = st.checkbox("Regresar al punto inicial", value=True)
        if st.button("Calcular Ruta Garzabus", use_container_width=True):
            if not graph:
                st.error("No se pudo calcular el recorrido porque el grafo no está disponible.")
            else:
                stops = [stop.strip() for stop in stops_text.splitlines() if stop.strip()]
                if len(stops) < 2:
                    st.error("Ingresa al menos dos paradas para calcular el Garzabus.")
                else:
                    try:
                        stop_coords = [resolve_address(stop, area) for stop in stops]
                        stop_nodes = [
                            routing.nearest_node(graph, lat, lon)
                            for lat, lon in stop_coords
                        ]
                        segments_nodes = routing.solve_nearest_neighbor_tour(
                            graph,
                            stop_nodes,
                            "travel_time",
                            max_speed_mps,
                            return_to_start=return_to_start,
                        )
                        route_segments_coords: List[List[Tuple[float, float]]] = []
                        total_length_m = 0.0
                        total_time_s = 0.0
                        for segment in segments_nodes:
                            segment_metrics = routing.compute_route_metrics(
                                graph, segment, minimize_key="travel_time"
                            )
                            total_length_m += segment_metrics["length_m"]
                            total_time_s += segment_metrics["travel_time_s"]
                            route_segments_coords.append(
                                routing.route_nodes_to_coordinates(graph, segment)
                            )

                        route_segments = route_segments_coords
                        markers = []
                        for idx, (stop, coords_pair) in enumerate(zip(stops, stop_coords), start=1):
                            markers.append(
                                {
                                    "location": list(coords_pair),
                                    "tooltip": f"Parada {idx}",
                                    "popup": stop,
                                    "color": "blue" if idx > 1 else "green",
                                }
                            )
                        if return_to_start and stops:
                            markers.append(
                                {
                                    "location": list(stop_coords[0]),
                                    "tooltip": "Cierre de ruta",
                                    "popup": stops[0],
                                    "color": "red",
                                }
                            )

                        fuel_cost = estimate_fuel_cost(total_length_m)
                        st.session_state["route_display"] = {
                            "segments": route_segments,
                            "markers": markers,
                            "summary": None,
                        }
                        st.session_state["garzabus_metrics"] = {
                            "distance": format_distance(total_length_m),
                            "time": format_duration(total_time_s),
                            "fuel": f"$ {fuel_cost:,.2f}",
                        }
                    except Exception as garza_error:
                        st.error(f"Error al calcular la Ruta del Garzabus: {garza_error}")
summary_to_show = st.session_state["route_display"]["summary"]
if summary_to_show:
    route_summary_placeholder.markdown(summary_to_show)
else:
    route_summary_placeholder.empty()

if st.session_state["garzabus_metrics"]:
    metrics = st.session_state["garzabus_metrics"]
    with garzabus_metrics_container:
        col1, col2, col3 = st.columns(3)
        col1.metric("Distancia Total", metrics["distance"])
        col2.metric("Tiempo Estimado", metrics["time"])
        col3.metric("Costo de Gasolina Est.", metrics["fuel"])

display_segments = st.session_state["route_display"]["segments"]
display_markers = st.session_state["route_display"]["markers"] or []

if display_segments:
    fmap = build_map(tuple(graph_center), route_segments=display_segments, markers=display_markers)
else:
    fmap = build_map(tuple(graph_center))

st_folium(fmap, height=680, width=None)
