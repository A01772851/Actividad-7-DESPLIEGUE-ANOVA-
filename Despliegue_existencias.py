import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import matplotlib.pyplot as plt

# Configuración de página con diseño personalizado
st.set_page_config(page_title="Actividad 7 - Despliegue ANOVA", layout="wide", page_icon="📊")

######################################################
# 1 y 2. Carga y Selección de Archivo/Hoja
######################################################
@st.cache_data
def load_data():
    file_path = "Datos_Complementarios_CLV.xlsx"
    sheet_name = "Existencias_15_10_2026"
    
    # Lectura del archivo excel
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df

df_raw = load_data()

st.title("📊 Actividad 7: Despliegue ANOVA (Socioformador CLV)")

######################################################
# 3. Preprocesamiento de Datos (Nulos, Outliers, Transformaciones)
######################################################
st.sidebar.header("⚙️ Configuración y Preprocesamiento")

# A. Manejo de Nulos
df = df_raw.copy()
df = df.bfill().ffill()

# Selección de Columnas según su tipo
cols_cat = ['articulo', 'nombre_articulo', 'marca', 'almacen']
cols_num = ['existencias', 'valor_inventario']

# B. Detección y Tratamiento de Outliers (Método IQR)
aplicar_outliers = st.sidebar.checkbox("Remover Outliers (Método IQR)", value=False)
if aplicar_outliers:
    var_outlier = st.sidebar.selectbox("Variable para filtrar outliers:", cols_num)
    Q1 = df[var_outlier].quantile(0.25)
    Q3 = df[var_outlier].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    df = df[(df[var_outlier] >= limite_inferior) & (df[var_outlier] <= limite_superior)]

# C. Transformaciones Numéricas
aplicar_transf = st.sidebar.checkbox("Aplicar Transformación Logarítmica log(x+1)", value=False)
if aplicar_transf:
    for col in cols_num:
        df[f"{col}_log"] = np.log1p(np.maximum(0, df[col]))
    cols_num_analisis = [f"{col}_log" for col in cols_num]
else:
    cols_num_analisis = cols_num

# Pestañas de Navegación principales
tab_prep, tab_anova_1, tab_anova_multi, tab_bonus = st.tabs([
    "🛠️ Preprocesamiento & Resumen", 
    "📈 ANOVA Unifactorial", 
    "🔀 ANOVA Multifactorial & Interacción", 
    "🎨 Visualizaciones Bonus"
])

######################################################
# Vista 1: Resumen de Preprocesamiento
######################################################
with tab_prep:
    st.subheader("📌 Datos Preprocesados")
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Total Registros", len(df))
    col_m2.metric("Variables Categóricas", len(cols_cat))
    col_m3.metric("Variables Numéricas", len(cols_num_analisis))
    
    st.dataframe(df.head(10), use_container_width=True)

######################################################
# 4A. ANOVA Unifactorial
######################################################
with tab_anova_1:
    st.header("1️⃣ Análisis de Varianza de Un Factor (ANOVA)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        var_y = st.selectbox("Variable Respuesta (Cuantitativa Y)", options=cols_num_analisis, key="a1_y")
        var_x = st.selectbox("Factor (Categoría X)", options=cols_cat, index=3, key="a1_x")
        
    grupos = [grupo[var_y].dropna() for _, grupo in df.groupby(var_x)]
    
    if len(grupos) > 1:
        f_val, p_val = stats.f_oneway(*grupos)
        
        with col1:
            st.markdown("### Resultados")
            st.metric("F-Valor", f"{f_val:.4f}")
            st.metric("p-Valor", f"{p_val:.6f}")
            
            if p_val < 0.05:
                st.success("✅ **Diferencia Significativa**: Las medias varían según las categorías (p < 0.05).")
            else:
                st.warning("⚠️ **Sin Diferencia Significativa**: No hay evidencia suficiente de variación entre grupos (p >= 0.05).")
                
        with col2:
            st.markdown("### Distribución por Categoría (Boxplot)")
            fig_box = px.box(
                df, x=var_x, y=var_y, color=var_x, 
                title=f"Distribución de {var_y} por {var_x}",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_box, use_container_width=True)

######################################################
# 4B. ANOVA Multifactorial y Gráficas de Interacción
######################################################
with tab_anova_multi:
    st.header("2️⃣ ANOVA Multifactorial (Factorial con Interacción)")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        var_y_m = st.selectbox("Variable Respuesta (Y)", options=cols_num_analisis, key="am_y")
    with col_f2:
        factor_1 = st.selectbox("Primer Factor (X1)", options=cols_cat, index=3, key="am_x1")
    with col_f3:
        factor_2 = st.selectbox("Segundo Factor (X2)", options=[c for c in cols_cat if c != factor_1], index=1, key="am_x2")

    # Ajuste de Modelo OLS para ANOVA Factorial
    formula = f"{var_y_m} ~ C({factor_1}) + C({factor_2}) + C({factor_1}):C({factor_2})"
    
    try:
        modelo = ols(formula, data=df).fit()
        tabla_anova = sm.stats.anova_lm(modelo, typ=2)
        
        st.subheader("📋 Tabla ANOVA Multifactorial")
        st.dataframe(tabla_anova.style.highlight_null(color="transparent"), use_container_width=True)
        
        st.subheader("📈 Gráfica de Interacción")
        # Agrupación para promedios de interacción
        df_inter = df.groupby([factor_1, factor_2])[var_y_m].mean().reset_index()
        
        fig_inter = px.line(
            df_inter, 
            x=factor_1, 
            y=var_y_m, 
            color=factor_2, 
            markers=True,
            title=f"Efecto de Interacción: {factor_1} vs {factor_2} sobre {var_y_m}",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_inter, use_container_width=True)
        
    except Exception as e:
        st.error(f"No se pudo calcular la interacción con los factores seleccionados (posible falta de combinaciones completas): {e}")

######################################################
# 5. Bonus: Widgets, Heatmaps, Color y Diseño Adicional
######################################################
with tab_bonus:
    st.header("🎨 Visualizaciones Adicionales (Bonus)")
    
    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
        st.subheader("🔥 Heatmap de Promedios por Categoría")
        var_heatmap_y = st.selectbox("Métrica Numérica", options=cols_num_analisis, key="hm_y")
        pivot_data = df.pivot_table(index=cols_cat[3], columns=cols_cat[2], values=var_heatmap_y, aggfunc='mean').fillna(0)
        
        fig_hm = px.imshow(
            pivot_data, 
            labels=dict(x="Marca", y="Almacén", color="Media"),
            title=f"Matriz de Calor: {var_heatmap_y} (Almacén vs Marca)",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_hm, use_container_width=True)
        
    with col_b2:
        st.subheader("📊 Tabla Comparativa por Grupos")
        resumen_tabla = df.groupby(cols_cat[3])[cols_num_analisis].agg(['mean', 'std', 'count']).reset_index()
        st.dataframe(resumen_tabla, use_container_width=True)