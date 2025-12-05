import streamlit as st
import numpy as np
from fractions import Fraction
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Solver PL", page_icon="📊")

# -------------------------------------------------------
# CSS PERSONALIZADO CON ANIMACIONES MEJORADO
# -------------------------------------------------------
st.markdown("""
<style>
    /* Estilos generales */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        animation: fadeIn 1s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Títulos principales - MÁS VISIBLE */
    .main-title {
        font-size: 3rem !important;
        font-weight: 800 !important;
        text-align: center !important;
        background: linear-gradient(90deg, #667eea, #764ba2, #FF4081) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2) !important;
        margin-bottom: 2rem !important;
        padding-bottom: 15px !important;
        position: relative !important;
        letter-spacing: 1px !important;
    }
    
    .main-title:after {
        content: '' !important;
        position: absolute !important;
        bottom: 0 !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: 200px !important;
        height: 5px !important;
        background: linear-gradient(90deg, #667eea, #764ba2, #FF4081) !important;
        border-radius: 3px !important;
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.4) !important;
    }
    
    h2, h3 {
        color: #2c3e50;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1.5rem;
        position: relative;
        padding-bottom: 10px;
    }
    
    h2:after, h3:after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 2px;
    }
    
    /* Tarjetas de entrada */
    .stTextInput > div > div, .stNumberInput > div > div {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 5px;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div:hover, .stNumberInput > div > div:hover {
        border-color: #667eea;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
        transform: translateY(-2px);
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #764ba2;
        box-shadow: 0 5px 15px rgba(118, 75, 162, 0.2);
    }
    
    /* Botones */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 14px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 18px;
        font-weight: 600;
        margin: 10px 2px;
        cursor: pointer;
        border-radius: 25px;
        transition: all 0.4s ease;
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-2px);
    }
    
    /* Pestañas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
        padding: 10px;
        border-radius: 15px;
        margin-bottom: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        color: #667eea;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Tablas personalizadas */
    .custom-table {
        background: white;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
        animation: slideIn 0.6s ease-out;
        border: 1px solid #e0e0e0;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .custom-table table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .custom-table th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 16px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1em;
        border: none;
    }
    
    .custom-table td {
        padding: 14px;
        text-align: center;
        border-bottom: 1px solid #f0f0f0;
        transition: background-color 0.3s ease;
        border: none;
    }
    
    .custom-table tr:hover td {
        background-color: #f8f9ff;
    }
    
    .custom-table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    
    /* Estilos para el gráfico */
    .graph-container {
        background: white;
        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
        animation: popIn 0.8s ease-out;
        border: 1px solid #e0e0e0;
    }
    
    @keyframes popIn {
        0% { opacity: 0; transform: scale(0.9); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    /* Contenedores de resultados */
    .result-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #eef1ff 100%);
        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.1);
        animation: slideUp 0.6s ease-out;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .result-card h4 {
        color: #2c3e50;
        font-size: 1.3em;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .result-card h4:before {
        content: '✓';
        background: #667eea;
        color: white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
    }
    
    /* SEPARADORES ELEGANTES - MÁS FINOS */
    .elegant-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
        margin: 30px 0;
        border-radius: 1px;
        opacity: 0.6;
    }
    
    .thin-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
        margin: 20px 0;
        border: none;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p {
        color: white !important;
    }
    
    /* Mensajes de éxito y error */
    .stAlert {
        border-radius: 15px;
        border: none;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(102, 126, 234, 0); }
        100% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0); }
    }
    
    /* Scrollbar personalizada */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Grid de entrada */
    .input-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
    
    /* Animaciones para números */
    .number-animation {
        animation: countUp 1s ease-out;
    }
    
    @keyframes countUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 5px 12px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
        margin: 0 5px;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }
    
    /* Tabla de comparación mejorada */
    .comparison-table {
        background: white;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
        border: 1px solid #e0e0e0;
    }
    
    .comparison-table table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .comparison-table th {
        background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%);
        color: white;
        padding: 18px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1em;
        border: none;
    }
    
    .comparison-table td {
        padding: 16px;
        text-align: center;
        border-bottom: 1px solid #f0f0f0;
        font-size: 1.1em;
        border: none;
    }
    
    .comparison-table tr:last-child td {
        border-bottom: none;
    }
    
    .comparison-table tr:hover td {
        background-color: #f9fff9;
    }
    
    /* Mensajes de resultado */
    .success-message {
        background: linear-gradient(135deg, #4CAF5020 0%, #8BC34A20 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        border-left: 5px solid #4CAF50;
        text-align: center;
        border: 1px solid rgba(76, 175, 80, 0.2);
    }
    
    .warning-message {
        background: linear-gradient(135deg, #FF980020 0%, #FF572220 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        border-left: 5px solid #FF9800;
        text-align: center;
        border: 1px solid rgba(255, 152, 0, 0.2);
    }
    
    /* Ajuste para las secciones blancas grandes */
    section.main {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 25px;
        padding: 30px;
        margin: 20px;
        box-shadow: 0 25px 70px rgba(0, 0, 0, 0.25);
    }
    
    /* Fondo para contenido principal */
    .st-emotion-cache-1v0mbdj {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 25px;
        padding: 30px;
        margin: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Título principal MÁS VISIBLE
st.markdown('<h1 class="main-title">📊 CALCULADORA DE PROGRAMACIÓN LINEAL AVANZADO</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="main-title">📊 Diego Alejandro Ramirez Russi </h3>', unsafe_allow_html=True)
st.markdown('<h3 class="main-title">📊 Jhoan Sebastian Saldarriaga</h3>', unsafe_allow_html=True)
st.markdown('<h3 class="main-title">📊 Ingrid Lorena Rubio</h3>', unsafe_allow_html=True)

# -------------------------------------------------------
# MANEJO DE ESTADO
# -------------------------------------------------------
if "generar_modelo" not in st.session_state:
    st.session_state.generar_modelo = False

if "datos_ingresados" not in st.session_state:
    st.session_state.datos_ingresados = False

# -------------------------------------------------------
# FUNCIONES PARA MÉTODO GRÁFICO (ORIGINAL)
# -------------------------------------------------------
def eval_fraction(expr):
    expr = expr.strip()
    if "/" in expr:
        num, den = expr.split("/")
        return float(num) / float(den)
    return float(expr)

def coef_for_var_term(term, var_char):
    before, after = term.split(var_char)
    before = before.strip()
    after = after.strip()
    
    if before in ("", "+"):
        numer = 1.0
    elif before == "-":
        numer = -1.0
    else:
        numer = eval_fraction(before)
    
    if after:
        numer *= eval_fraction(after.replace("/", ""))
    
    return numer

def parse_term(term):
    t = term.strip()
    if "x" in t:
        return ("x", coef_for_var_term(t, "x"))
    elif "y" in t:
        return ("y", coef_for_var_term(t, "y"))
    else:
        return ("c", eval_fraction(t))

def parse_side(side):
    s = side.replace("-", "+-")
    if s.startswith("+-"):
        s = s[1:]
    parts = s.split("+")
    a = b = c = 0
    for p in parts:
        if p.strip():
            kind, val = parse_term(p)
            if kind == "x": a += val
            elif kind == "y": b += val
            else: c += val
    return a, b, c

def parse_constraint(expr):
    expr = expr.replace(" ", "")
    if ">=" in expr: sign = ">="
    elif "<=" in expr: sign = "<="
    elif "=" in expr: sign = "="
    else: raise ValueError("Debe contener <=, >= o =")
    
    left, right = expr.split(sign)
    ax1, by1, c1 = parse_side(left)
    ax2, by2, c2 = parse_side(right)
    
    return (ax1-ax2, by1-by2, sign, c2-c1)

# -------------------------------------------------------
# FUNCIONES PARA DOS FASES (ORIGINAL)
# -------------------------------------------------------
def to_frac(x):
    """Convierte un número a fracción solo para mostrar."""
    try:
        return str(Fraction(x).limit_denominator())
    except:
        return x

def calcular_Zj(tabla, base_idx, c):
    m, n = tabla.shape
    n -= 1
    Z = np.zeros(n)
    for j in range(n):
        for i, b in enumerate(base_idx):
            Z[j] += c[b] * tabla[i, j]
    return Z

def mostrar_tabla(tabla, base_idx, nombres_vars, c):
    m, n = tabla.shape
    n -= 1
    Z = calcular_Zj(tabla, base_idx, c)
    Cj_Zj = np.array([c[j] - Z[j] for j in range(n)])
    
    # Usar el estilo CSS personalizado para la tabla
    html = f"""
    <div class="custom-table">
        <table>
            <thead>
                <tr>
                    <th>VB</th>
    """
    
    for j in range(n):
        html += f'<th>{nombres_vars[j]}</th>'
    html += '<th>Solución</th></tr></thead><tbody>'
    
    for i, b in enumerate(base_idx):
        html += f'<tr><td><strong>{nombres_vars[b]}</strong></td>'
        for j in range(n):
            html += f'<td>{to_frac(tabla[i, j])}</td>'
        html += f'<td><strong>{to_frac(tabla[i, -1])}</strong></td></tr>'
    
    html += f'<tr style="background-color: #f0f8ff;"><td><strong>Z</strong></td>'
    for j in range(n):
        html += f'<td>{to_frac(Z[j])}</td>'
    html += '<td></td></tr>'
    
    html += f'<tr style="background-color: #fff0f0;"><td><strong>Cj-Zj</strong></td>'
    for j in range(n):
        color = "color: green;" if Cj_Zj[j] >= 0 else "color: red;"
        html += f'<td style="{color}"><strong>{to_frac(Cj_Zj[j])}</strong></td>'
    html += '<td></td></tr>'
    
    html += '</tbody></table></div>'
    
    st.markdown(html, unsafe_allow_html=True)

def simplex_pivote(tabla, base_idx, c, tipo):
    m, n = tabla.shape
    n -= 1
    
    Z = calcular_Zj(tabla, base_idx, c)
    Cj_Zj = c[:n] - Z
    
    if tipo == "max":
        col_entrada = np.argmax(Cj_Zj)
        if Cj_Zj[col_entrada] <= 0:
            return None, None, tabla
    else:
        col_entrada = np.argmin(Cj_Zj)
        if Cj_Zj[col_entrada] >= 0:
            return None, None, tabla
    
    ratios = []
    for i in range(m):
        if tabla[i, col_entrada] > 0:
            ratios.append(tabla[i, -1] / tabla[i, col_entrada])
        else:
            ratios.append(np.inf)
    
    fila_salida = np.argmin(ratios)
    if ratios[fila_salida] == np.inf:
        st.error("Problema no acotado")
        return None, None, tabla
    
    pivote = tabla[fila_salida, col_entrada]
    tabla[fila_salida, :] /= pivote
    
    for i in range(m):
        if i != fila_salida:
            tabla[i, :] -= tabla[i, col_entrada] * tabla[fila_salida, :]
    
    base_idx[fila_salida] = col_entrada
    return fila_salida, col_entrada, tabla

# -------------------------------------------------------
# INICIALIZACIÓN DE VARIABLES GLOBALES
# -------------------------------------------------------
# Inicializar restricciones como lista vacía
restricciones = []
coef_obj = []

# -------------------------------------------------------
# INTERFAZ DE ENTRADA ÚNICA
# -------------------------------------------------------
st.markdown("### 🎯 INGRESO DE DATOS DEL MODELO")

# Configuración básica
col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("📈 VARIABLES DEL MODELO")
    num_vars = st.number_input("Número de Variables (máx. 20)", min_value=1, max_value=20, step=1, key="vars")
    st.markdown('</div>', unsafe_allow_html=True)
    
with col2:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("🎯 OBJETIVO")
    tipo = st.selectbox("Tipo de Optimización:", ["Maximizar", "Minimizar"], key="tipo_obj")
    st.markdown('</div>', unsafe_allow_html=True)

# Función objetivo
st.markdown('<div class="result-card">', unsafe_allow_html=True)
st.subheader("📊 FUNCIÓN OBJETIVO")
st.markdown("Ingrese los coeficientes para cada variable:")

coef_obj = []  # Reinicializar coef_obj
cols = st.columns(min(num_vars, 6))

for i in range(num_vars):
    with cols[i % 6]:
        val = st.text_input(f"X{i+1}", "0", key=f"obj_{i}")
        try:
            coef_obj.append(float(val))
        except:
            coef_obj.append(0.0)
st.markdown('</div>', unsafe_allow_html=True)

# Configuración de restricciones
st.markdown('<div class="result-card">', unsafe_allow_html=True)
st.subheader("🔗 RESTRICCIONES")
num_rest = st.number_input("Número de Restricciones (máx. 50)", min_value=1, max_value=50, step=1, key="rests")
st.markdown('</div>', unsafe_allow_html=True)

# Restricciones
restricciones = []  # Reinicializar restricciones
for i in range(num_rest):
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(f"**Restricción {i+1}:**")
    cols = st.columns(num_vars + 2)
    
    coefs = []
    for j in range(num_vars):
        with cols[j]:
            val = st.text_input(f"X{j+1}", "0", key=f"r{i}_{j}")
            try:
                coefs.append(float(val))
            except:
                coefs.append(0.0)
    
    with cols[num_vars]:
        signo = st.selectbox("Signo", ["<=", ">=", "="], key=f"s{i}")
    
    with cols[num_vars + 1]:
        rhs = st.text_input("Valor", "0", key=f"b{i}")
    
    try:
        restricciones.append((np.array(coefs), signo, float(rhs)))
    except:
        restricciones.append((np.array(coefs), signo, 0.0))
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'<div class="result-card"><h4>✅ CONDICIÓN DE NO NEGATIVIDAD</h4><p>X₁, X₂, ..., X{num_vars} ≥ 0</p></div>', unsafe_allow_html=True)

# Botón para resolver
st.markdown('<div class="elegant-divider"></div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if st.button("🚀 RESOLVER POR AMBOS MÉTODOS", type="primary"):
        st.session_state.datos_ingresados = True
        st.rerun()

# -------------------------------------------------------
# RESOLUCIÓN POR AMBOS MÉTODOS
# -------------------------------------------------------
if st.session_state.datos_ingresados:
    st.markdown('<div class="elegant-divider"></div>', unsafe_allow_html=True)
    st.markdown("## 📋 RESULTADOS POR AMBOS MÉTODOS")
    
    # Verificar datos
    if len(coef_obj) != num_vars:
        st.error("Error en los coeficientes de la función objetivo")
        st.stop()
    
    if len(restricciones) != num_rest:
        st.error("Error en las restricciones")
        st.stop()
    
    # =======================================================
    # MÉTODO DE DOS FASES
    # =======================================================
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown("### 🔄 MÉTODO DE DOS FASES")
    
    tipo_simplex = "max" if tipo == "Maximizar" else "min"
    
    nombres_vars = [f"X{i+1}" for i in range(num_vars)]
    nombres_vars += [f"S{i+1}" for i in range(num_rest)]
    num_art = sum(1 for _, s, _ in restricciones if s in [">=", "="])
    nombres_vars += [f"A{i+1}" for i in range(num_art)]
    
    total_cols = num_vars + num_rest + num_art
    A = np.zeros((len(restricciones), total_cols))
    b = np.zeros(len(restricciones))
    base_idx = []
    art_idx = num_vars + num_rest
    art_list = []
    
    for i, (coefs, signo, rhs) in enumerate(restricciones):
        A[i, :num_vars] = coefs
        b[i] = rhs
        if signo == "<=":
            A[i, num_vars + i] = 1
            base_idx.append(num_vars + i)
        elif signo == ">=":
            A[i, num_vars + i] = -1
            A[i, art_idx] = 1
            base_idx.append(art_idx)
            art_list.append(art_idx)
            art_idx += 1
        elif signo == "=":
            A[i, art_idx] = 1
            base_idx.append(art_idx)
            art_list.append(art_idx)
            art_idx += 1
    
    c1 = np.zeros(total_cols)
    for idx in art_list:
        c1[idx] = 1
    
    tabla = np.hstack([A, b.reshape(-1, 1)])
    
    with st.expander("📊 VER TABLAS DEL MÉTODO DE DOS FASES", expanded=True):
        tab1, tab2 = st.tabs(["🎯 FASE 1", "🚀 FASE 2"])
        
        with tab1:
            st.subheader("TABLA INICIAL FASE 1")
            mostrar_tabla(tabla, base_idx, nombres_vars, c1)
            
            iter_num = 1
            while True:
                fila_salida, col_entrada, tabla = simplex_pivote(tabla, base_idx, c1, "min")
                if fila_salida is None:
                    break
                st.subheader(f"ITERACIÓN {iter_num} FASE 1")
                mostrar_tabla(tabla, base_idx, nombres_vars, c1)
                iter_num += 1
        
        # Fase 2
        with tab2:
            keep_cols = [i for i in range(total_cols) if i not in art_list]
            A2 = tabla[:, keep_cols]
            b2 = tabla[:, -1]
            
            base_idx2 = []
            for i, b_idx in enumerate(base_idx):
                if b_idx in keep_cols:
                    base_idx2.append(keep_cols.index(b_idx))
                else:
                    for k in range(len(keep_cols)):
                        if k not in base_idx2:
                            base_idx2.append(k)
                            break
            
            nombres_vars2 = [nombres_vars[i] for i in keep_cols]
            
            c2 = np.zeros(len(keep_cols))
            c2[:num_vars] = coef_obj
            
            tabla2 = np.hstack([A2, b2.reshape(-1, 1)])
            
            st.subheader("TABLA INICIAL FASE 2")
            mostrar_tabla(tabla2, base_idx2, nombres_vars2, c2)
            
            iter_num = 1
            while True:
                fila_salida, col_entrada, tabla2 = simplex_pivote(tabla2, base_idx2, c2, tipo_simplex)
                if fila_salida is None:
                    break
                st.subheader(f"ITERACIÓN {iter_num} FASE 2")
                mostrar_tabla(tabla2, base_idx2, nombres_vars2, c2)
                iter_num += 1
    
    # Solución final del método de dos fases
    sol = np.zeros(len(nombres_vars2))
    for i, idx in enumerate(base_idx2):
        sol[idx] = tabla2[i, -1]
    
    valor_Z = np.dot(c2, sol)
    
    # Mostrar solución con estilo
    st.markdown("### ✅ SOLUCIÓN ÓPTIMA (MÉTODO DE DOS FASES)")
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%); padding: 20px; border-radius: 15px; margin: 20px 0; border: 1px solid rgba(102, 126, 234, 0.2);">
        <h4 style="color: #2c3e50; margin-bottom: 15px;">📊 VARIABLES DE DECISIÓN:</h4>
    """, unsafe_allow_html=True)
    
    for i in range(num_vars):
        st.markdown(f"""
        <div style="background: white; padding: 10px 20px; margin: 8px 0; border-radius: 10px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); border: 1px solid #f0f0f0;">
            <span style="font-weight: bold; color: #667eea;">X{i+1}</span>
            <span style="font-weight: bold; font-size: 1.2em; color: #2c3e50;">{to_frac(sol[i])}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; margin-top: 20px; border-radius: 10px; text-align: center; border: none;">
            <h4 style="color: white; margin: 0;">
                VALOR ÓPTIMO Z ({tipo.upper()}): <span style="font-size: 1.5em;">{to_frac(valor_Z)}</span>
            </h4>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # =======================================================
    # MÉTODO GRÁFICO (solo si hay 2 variables)
    # =======================================================
    st.markdown('<div class="elegant-divider"></div>', unsafe_allow_html=True)
    
    if num_vars == 2:
        st.markdown('<div class="graph-container">', unsafe_allow_html=True)
        st.markdown("### 📈 MÉTODO GRÁFICO")
        
        # Convertir los datos al formato del método gráfico
        try:
            # Crear restricciones para el método gráfico
            restricciones_grafico = []
            
            # Restricciones de no negatividad
            restricciones_grafico.append((1, 0, ">=", 0))   # x ≥ 0
            restricciones_grafico.append((0, 1, ">=", 0))   # y ≥ 0
            
            # Convertir restricciones del formato matricial al de texto
            for i, (coefs, signo, rhs) in enumerate(restricciones):
                a, b = coefs[0], coefs[1]
                restricciones_grafico.append((a, b, signo, rhs))
            
            # Funciones para método gráfico
            def solve_intersection(l1, l2):
                a1, b1, _, c1 = l1
                a2, b2, _, c2 = l2
                det = a1*b2 - a2*b1
                if abs(det) < 1e-9:
                    return None
                x = (c1*b2 - c2*b1) / det
                y = (a1*c2 - a2*c1) / det
                return (x, y)
            
            def satisfies_all(p, constraints):
                x, y = p
                for a, b, sign, c in constraints:
                    lhs = a*x + b*y
                    if sign == "<=" and lhs > c: return False
                    if sign == ">=" and lhs < c: return False
                    if sign == "=" and abs(lhs-c) > 1e-7: return False
                return True
            
            def convex_hull(points):
                pts = sorted(set(points))
                if len(pts) <= 1: return pts
                
                def cross(o, a, b):
                    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
                
                lower, upper = [], []
                for p in pts:
                    while len(lower)>=2 and cross(lower[-2], lower[-1], p)<=0:
                        lower.pop()
                    lower.append(p)
                
                for p in reversed(pts):
                    while len(upper)>=2 and cross(upper[-2], upper[-1], p)<=0:
                        upper.pop()
                    upper.append(p)
                
                return lower[:-1] + upper[:-1]
            
            # Calcular puntos factibles
            candidates = [(0, 0)]
            for i in range(len(restricciones_grafico)):
                for j in range(i+1, len(restricciones_grafico)):
                    p = solve_intersection(restricciones_grafico[i], restricciones_grafico[j])
                    if p: candidates.append(p)
            
            feasible = [p for p in candidates if satisfies_all(p, restricciones_grafico)]
            
            if feasible:
                def Z(p): return coef_obj[0]*p[0] + coef_obj[1]*p[1]
                optimal_point = max(feasible, key=Z) if tipo == "Maximizar" else min(feasible, key=Z)
                
                hull = convex_hull(feasible)
                
                # Crear gráfico con estilo mejorado
                fig = go.Figure()
                x_vals = np.linspace(0, max(p[0] for p in feasible)+5, 400)
                
                # Dibujar restricciones (sin incluir x≥0, y≥0)
                visibles = restricciones_grafico[2:]
                
                colors = ['#667eea', '#764ba2', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']
                
                for idx, (a,b,sign,c) in enumerate(visibles):
                    color = colors[idx % len(colors)]
                    if abs(b) > 1e-9:
                        y_vals = (c - a*x_vals)/b
                        fig.add_trace(go.Scatter(
                            x=x_vals, y=y_vals,
                            mode="lines",
                            name=f"R{idx+1}: {a}X₁ + {b}X₂ {sign} {c}",
                            line=dict(color=color, width=3, dash='dash'),
                            showlegend=True
                        ))
                    else:
                        x0 = c/a
                        fig.add_trace(go.Scatter(
                            x=[x0, x0], y=[0, max(p[1] for p in feasible)+5],
                            mode="lines",
                            name=f"R{idx+1}: X₁ {sign} {c/a}",
                            line=dict(color=color, width=3, dash='dash'),
                            showlegend=True
                        ))
                
                if len(hull) >= 3:
                    hx = [p[0] for p in hull] + [hull[0][0]]
                    hy = [p[1] for p in hull] + [hull[0][1]]
                    
                    # Región factible con gradiente
                    fig.add_trace(go.Scatter(
                        x=hx, y=hy,
                        fill="toself",
                        fillcolor='rgba(102, 126, 234, 0.3)',
                        line=dict(color='rgba(102, 126, 234, 0.8)', width=2),
                        name="Región Factible",
                        opacity=0.6,
                        showlegend=True
                    ))
                
                # Punto óptimo con animación
                fig.add_trace(go.Scatter(
                    x=[optimal_point[0]], y=[optimal_point[1]],
                    mode="markers+text",
                    text=["PUNTO ÓPTIMO"],
                    textposition="top center",
                    marker=dict(
                        size=20,
                        color='#FF4081',
                        symbol='star',
                        line=dict(color='white', width=2)
                    ),
                    name="Punto Óptimo",
                    showlegend=True
                ))
                
                # Líneas de nivel para la función objetivo
                z_values = np.linspace(Z(feasible[0]), Z(optimal_point), 5)
                for z_val in z_values:
                    if abs(coef_obj[1]) > 1e-9:
                        y_line = (z_val - coef_obj[0]*x_vals)/coef_obj[1]
                        fig.add_trace(go.Scatter(
                            x=x_vals, y=y_line,
                            mode="lines",
                            line=dict(color='rgba(255, 64, 129, 0.2)', width=1, dash='dot'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                
                fig.update_layout(
                    xaxis_title="X₁",
                    yaxis_title="X₂",
                    xaxis=dict(
                        range=[0, max(p[0] for p in feasible)+5],
                        gridcolor='rgba(0,0,0,0.1)',
                        zerolinecolor='rgba(0,0,0,0.3)'
                    ),
                    yaxis=dict(
                        range=[0, max(p[1] for p in feasible)+5],
                        gridcolor='rgba(0,0,0,0.1)',
                        zerolinecolor='rgba(0,0,0,0.3)'
                    ),
                    title=dict(
                        text="📈 SOLUCIÓN GRÁFICA DEL PROBLEMA",
                        font=dict(size=24, color='#2c3e50')
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    hovermode='closest',
                    width=1000,
                    height=700,
                    legend=dict(
                        bgcolor='rgba(255,255,255,0.9)',
                        bordercolor='rgba(0,0,0,0.1)',
                        borderwidth=1
                    )
                )
                
                # Mostrar gráfico
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar resultados del método gráfico con estilo
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%); padding: 20px; border-radius: 15px; margin: 20px 0; border: 1px solid rgba(102, 126, 234, 0.2);">
                    <h4 style="color: #2c3e50; margin-bottom: 15px;">✅ SOLUCIÓN ÓPTIMA (MÉTODO GRÁFICO)</h4>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="📌 PUNTO ÓPTIMO (X₁, X₂)",
                        value=f"({optimal_point[0]:.4f}, {optimal_point[1]:.4f})",
                        delta="SOLUCIÓN ENCONTRADA"
                    )
                
                with col2:
                    st.metric(
                        label=f"🎯 VALOR Z ({tipo.upper()})",
                        value=f"{Z(optimal_point):.4f}",
                        delta="ÓPTIMO"
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Mostrar puntos factibles con estilo
                with st.expander("🔍 VER TODOS LOS PUNTOS FACTIBLES"):
                    cols = st.columns(3)
                    for i, point in enumerate(feasible):
                        with cols[i % 3]:
                            st.markdown(f"""
                            <div style="background: white; padding: 15px; margin: 10px 0; border-radius: 10px; box-shadow: 0 3px 10px rgba(0,0,0,0.1); text-align: center; border: 1px solid #f0f0f0;">
                                <div style="font-size: 0.8em; color: #667eea; margin-bottom: 5px;">PUNTO {i+1}</div>
                                <div style="font-size: 1.2em; font-weight: bold; color: #2c3e50;">
                                    ({point[0]:.2f}, {point[1]:.2f})
                                </div>
                                <div style="font-size: 0.9em; color: #666; margin-top: 5px;">
                                    Z = {Z(point):.2f}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            
            else:
                st.warning("No se encontró región factible con el método gráfico.")
        
        except Exception as e:
            st.error(f"Error en el método gráfico: {str(e)}")
            st.info("El método gráfico solo funciona correctamente para problemas con 2 variables.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="warning-message">
            <h4 style="color: #FF9800; margin: 0;">⚠️ INFORMACIÓN</h4>
            <p style="margin: 10px 0;">El método gráfico solo está disponible para problemas con <strong>2 variables</strong>.</p>
            <p style="margin: 0;">Tu problema tiene <strong>{}</strong> variables, por lo que solo se muestra la solución por el método de dos fases.</p>
        </div>
        """.format(num_vars), unsafe_allow_html=True)
    
    # =======================================================
    # COMPARACIÓN DE RESULTADOS (si aplica) - CORREGIDO
    # =======================================================
    if num_vars == 2 and 'optimal_point' in locals():
        st.markdown('<div class="elegant-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("### 📊 COMPARACIÓN DE RESULTADOS")
        
        # Resultados del método de dos fases
        sol_dos_fases = (sol[0], sol[1])
        Z_dos_fases = valor_Z
        
        # Resultados del método gráfico
        sol_grafico = optimal_point
        Z_grafico = Z(optimal_point)
        
        # Crear tabla comparativa con estilo mejorado
        st.markdown("""
        <div class="comparison-table">
            <table>
                <thead>
                    <tr>
                        <th>MÉTODO</th>
                        <th>X₁</th>
                        <th>X₂</th>
                        <th>VALOR Z</th>
                        <th>ESTADO</th>
                    </tr>
                </thead>
                <tbody>
        """, unsafe_allow_html=True)
        
        # Fila para Dos Fases
        st.markdown(f"""
            <tr>
                <td><strong>DOS FASES</strong></td>
                <td>{to_frac(sol_dos_fases[0])}</td>
                <td>{to_frac(sol_dos_fases[1])}</td>
                <td><strong>{to_frac(Z_dos_fases)}</strong></td>
                <td><span style="color: #4CAF50; font-size: 1.2em;">✓</span></td>
            </tr>
        """, unsafe_allow_html=True)
        
        # Fila para Gráfico
        st.markdown(f"""
            <tr>
                <td><strong>GRÁFICO</strong></td>
                <td>{sol_grafico[0]:.4f}</td>
                <td>{sol_grafico[1]:.4f}</td>
                <td><strong>{Z_grafico:.4f}</strong></td>
                <td><span style="color: #4CAF50; font-size: 1.2em;">✓</span></td>
            </tr>
        """, unsafe_allow_html=True)
        
        st.markdown("</tbody></table></div>", unsafe_allow_html=True)
        
        # Verificar si los resultados son similares
        diferencia = abs(Z_dos_fases - Z_grafico)
        if diferencia < 0.001:
            # CORREGIDO: Usar markdown en lugar de st.success con unsafe_allow_html
            st.markdown("""
            <div class="success-message">
                <h4 style="color: #4CAF50; margin: 0;">✅ COINCIDENCIA PERFECTA</h4>
                <p style="margin: 10px 0;">Ambos métodos coinciden perfectamente en la solución óptima.</p>
                <p style="margin: 0; font-size: 0.9em; color: #666;">Diferencia: {:.6f}</p>
            </div>
            """.format(diferencia), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-message">
                <h4 style="color: #FF9800; margin: 0;">⚠️ PEQUEÑA DIFERENCIA</h4>
                <p style="margin: 10px 0;">Hay una pequeña diferencia entre los métodos: {:.6f}</p>
                <p style="margin: 0; font-size: 0.9em; color: #666;">Esto puede deberse a aproximaciones numéricas.</p>
            </div>
            """.format(diferencia), unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # =======================================================
    # RESUMEN FINAL
    # =======================================================
    st.markdown('<div class="elegant-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown("### 📋 RESUMEN DEL MODELO")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 20px; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.05); border: 1px solid #f0f0f0;">
            <h4 style="color: #667eea; margin-bottom: 15px;">🎯 FUNCIÓN OBJETIVO</h4>
        """, unsafe_allow_html=True)
        
        if tipo == "Maximizar":
            obj_str = f"Max Z = "
        else:
            obj_str = f"Min Z = "
        
        terminos = []
        for i in range(num_vars):
            if coef_obj[i] != 0:
                signo = "+" if coef_obj[i] >= 0 else "-"
                valor = abs(coef_obj[i])
                terminos.append(f"{signo} {valor}X{i+1}")
        
        if terminos:
            # Eliminar el primer signo + si existe
            if terminos[0].startswith("+ "):
                terminos[0] = terminos[0][2:]
            obj_str += " ".join(terminos)
        else:
            obj_str += "0"
        
        st.markdown(f'<div style="font-size: 1.2em; padding: 10px; background: #f8f9ff; border-radius: 8px; text-align: center; border: 1px solid #e0e0e0;">{obj_str}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 20px; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.05); border: 1px solid #f0f0f0;">
            <h4 style="color: #764ba2; margin-bottom: 15px;">🔗 RESTRICCIONES</h4>
        """, unsafe_allow_html=True)
        
        for i, (coefs, signo, rhs) in enumerate(restricciones):
            terminos = []
            for j in range(num_vars):
                if coefs[j] != 0:
                    signo_coef = "+" if coefs[j] >= 0 or j == 0 else "-"
                    valor = abs(coefs[j])
                    if j == 0:
                        terminos.append(f"{valor}X{j+1}")
                    else:
                        terminos.append(f"{signo_coef} {valor}X{j+1}")
            
            if terminos:
                rest_str = " ".join(terminos)
            else:
                rest_str = "0"
            
            rest_str += f" {signo} {rhs}"
            st.markdown(f'<div style="padding: 8px 0; border-bottom: 1px solid #eee;">R{i+1}: {rest_str}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------
# INFORMACIÓN ADICIONAL EN SIDEBAR
# -------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.2);">
        <h2 style="color: white; text-align: center;">📊 SOLVER PL</h2>
        <p style="color: rgba(255,255,255,0.9); text-align: center;">Herramienta Avanzada de Programación Lineal</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.2);">
        <h3 style="color: white;">🚀 CARACTERÍSTICAS</h3>
        <ul style="color: rgba(255,255,255,0.9);">
            <li>Entrada única de datos</li>
            <li>Método de Dos Fases</li>
            <li>Método Gráfico (2 variables)</li>
            <li>Comparación automática</li>
            <li>Resultados en tiempo real</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.2);">
        <h3 style="color: white;">📝 INSTRUCCIONES</h3>
        <ol style="color: rgba(255,255,255,0.9);">
            <li>Ingresa número de variables</li>
            <li>Selecciona tipo de optimización</li>
            <li>Define la función objetivo</li>
            <li>Agrega las restricciones</li>
            <li>Haz clic en "Resolver"</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; color: rgba(255,255,255,0.7);">
        <small>© 2024 Solver PL - Herramienta Educativa</small>
    </div>

    """, unsafe_allow_html=True)


