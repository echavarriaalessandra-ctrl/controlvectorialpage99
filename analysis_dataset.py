import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import re

def load_and_preprocess_data(filepath):
    """
    Carga el archivo CSV y realiza el preprocesamiento inicial.
    """
    try:
        df = pd.read_csv(filepath, sep=';', engine='python')
    except FileNotFoundError:
        print(f"Error: El archivo no se encontró en {filepath}")
        return None

    # Limpiar nombres de columnas (eliminar espacios y caracteres especiales)
    df.columns = df.columns.str.replace('"', '').str.strip()
    df.rename(columns={'poverty_rate ': 'poverty_rate'}, inplace=True)

    # Convertir columnas numéricas con coma decimal a punto decimal
    cols_to_convert = ['poverty_rate', 'no_services_rate', 'wolbachia_prevalence']
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = df[col].str.replace(',', '.', regex=False).astype(float)

    # Convertir fechas
    df['report_date'] = pd.to_datetime(df['report_date'])
    df['collection_date'] = pd.to_datetime(df['collection_date'])

    # Asegurarse de que las columnas de confirmación sean numéricas
    for col in ['dengue_confirmed', 'zika_confirmed', 'chik_confirmed']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    return df

def calculate_risk_factors(df):
    """
    Calcula métricas agregadas por comuna para identificar factores de riesgo.
    """
    if df is None:
        return None

    # Calcular la incidencia de enfermedades por comuna
    # Sumar casos confirmados de las 3 enfermedades
    df['total_confirmed_cases'] = df['dengue_confirmed'] + df['zika_confirmed'] + df['chik_confirmed']

    # Agrupar por ciudad y comuna para obtener métricas agregadas
    comuna_summary = df.groupby(['city', 'comuna']).agg(
        total_cases=('total_confirmed_cases', 'sum'),
        avg_poverty_rate=('poverty_rate', 'mean'),
        avg_no_services_rate=('no_services_rate', 'mean'),
        avg_wolbachia_prevalence=('wolbachia_prevalence', 'mean'),
        total_mosquitoes_tested=('mosquitoes_tested', 'sum'),
        total_mosquitoes_wolbachia_pos=('mosquitoes_wolbachia_pos', 'sum')
    ).reset_index()

    # Calcular la prevalencia real de Wolbachia si hay datos de mosquitos
    comuna_summary['calculated_wolbachia_prevalence'] = comuna_summary.apply(
        lambda row: row['total_mosquitoes_wolbachia_pos'] / row['total_mosquitoes_tested']
        if row['total_mosquitoes_tested'] > 0 else 0, axis=1
    )
    # Usar la prevalencia calculada si es más precisa o si la original es NaN
    comuna_summary['final_wolbachia_prevalence'] = comuna_summary['avg_wolbachia_prevalence'].fillna(
        comuna_summary['calculated_wolbachia_prevalence']
    )
    # Si sigue siendo NaN (no hay datos de mosquitos), rellenar con 0 o la media global
    comuna_summary['final_wolbachia_prevalence'].fillna(0, inplace=True)

    # Normalizar variables para índice de riesgo
    comuna_summary['normalized_total_cases'] = (comuna_summary['total_cases'] - comuna_summary['total_cases'].min()) / (comuna_summary['total_cases'].max() - comuna_summary['total_cases'].min())
    comuna_summary['normalized_poverty_rate'] = (comuna_summary['avg_poverty_rate'] - comuna_summary['avg_poverty_rate'].min()) / (comuna_summary['avg_poverty_rate'].max() - comuna_summary['avg_poverty_rate'].min())
    comuna_summary['normalized_no_services_rate'] = (comuna_summary['avg_no_services_rate'] - comuna_summary['avg_no_services_rate'].min()) / (comuna_summary['avg_no_services_rate'].max() - comuna_summary['avg_no_services_rate'].min())
    
    # Invertir la prevalencia de Wolbachia para que menor prevalencia signifique mayor riesgo
    comuna_summary['inverted_wolbachia_prevalence'] = 1 - comuna_summary['final_wolbachia_prevalence']
    comuna_summary['normalized_inverted_wolbachia'] = (comuna_summary['inverted_wolbachia_prevalence'] - comuna_summary['inverted_wolbachia_prevalence'].min()) / (comuna_summary['inverted_wolbachia_prevalence'].max() - comuna_summary['inverted_wolbachia_prevalence'].min())

    # Calcular el índice de riesgo con pesos
    comuna_summary['risk_index'] = (
        comuna_summary['normalized_total_cases'] * 0.4 +
        comuna_summary['normalized_poverty_rate'] * 0.3 +
        comuna_summary['normalized_no_services_rate'] * 0.2 +
        comuna_summary['normalized_inverted_wolbachia'] * 0.1
    )

    # Ordenar por índice de riesgo descendente
    comuna_summary = comuna_summary.sort_values(by='risk_index', ascending=False)

    return comuna_summary

def show_and_save_top_risk_table(df_summary, top_n=10):
    """
    Muestra y guarda la tabla con las comunas de mayor riesgo combinado.
    """
    if df_summary is None:
        print("No hay datos para mostrar la tabla de riesgo.")
        return

    cols = ['city', 'comuna', 'total_cases', 'avg_poverty_rate', 'avg_no_services_rate', 'final_wolbachia_prevalence', 'risk_index']
    top_risk = df_summary[cols].head(top_n)

    print("\n--- Comunas con Mayor Riesgo Combinado (Top {}) ---".format(top_n))
    print(top_risk.to_string(index=False))

    # Guardar tabla como CSV para uso externo si quieres
    top_risk.to_csv('top_risk_comunas.csv', index=False)

def plot_disease_distribution_plotly(df):
    """
    Grafica la distribución total de casos por enfermedad usando Plotly.
    """
    if df is None:
        print("No hay datos para graficar distribución de casos por enfermedad.")
        return

    # Sumar casos totales por enfermedad
    total_dengue = df['dengue_confirmed'].sum()
    total_zika = df['zika_confirmed'].sum()
    total_chik = df['chik_confirmed'].sum()

    diseases = ['Dengue', 'Zika', 'Chikungunya']
    cases = [total_dengue, total_zika, total_chik]

    df_cases = pd.DataFrame({'Enfermedad': diseases, 'Número de Casos Confirmados': cases})

    fig = px.bar(df_cases, x='Enfermedad', y='Número de Casos Confirmados', 
                 title='Distribución de Casos por Enfermedad',
                 color='Enfermedad',
                 color_discrete_sequence=px.colors.qualitative.Vivid) 
    fig.show()
    fig.write_html("disease_cases_distribution.html")

def plot_cases_by_sex_plotly(df):
    """
    Grafica los casos confirmados por sexo para cada enfermedad usando Plotly.
    Las barras se mostrarán una al lado de la otra para cada sexo.
    """ 
    if df is None or 'sex' not in df.columns:
        print("No hay datos o columna 'sex' para graficar casos por sexo.")
        return

    sex_cases = df.groupby('sex')[['dengue_confirmed', 'zika_confirmed', 'chik_confirmed']].sum().reset_index()

    # Transformar para gráfico tipo barras apiladas (o agrupadas en este caso)
    sex_cases_melted = sex_cases.melt(id_vars='sex', var_name='disease', value_name='cases')

    fig = px.bar(sex_cases_melted, x='sex', y='cases', color='disease',
                 title='Casos Confirmados por Sexo por Enfermedad',
                 labels={'sex': 'Sexo', 'cases': 'Número de Casos Confirmados', 'disease': 'Enfermedad'},
                 barmode='group', 
                 color_discrete_sequence=px.colors.qualitative.Set2) 
    fig.show()
    fig.write_html("cases_by_sex.html")

def plot_disease_proportion_by_sex_pie_plotly(df):
    """
    Grafica la proporción de casos confirmados por enfermedad y sexo en un gráfico de torta de Plotly.
    Desagrega por: Dengue/Zika/Chikungunya x Hombres/Mujeres.
    """
    if df is None:
        print("No hay datos para graficar la proporción de casos por enfermedad y sexo.")
        return

    # Asegurar que la columna 'sex' esté limpia (por ejemplo, 'M' para hombres, 'F' para mujeres)
    df['sex'] = df['sex'].str.upper().str.strip()  # Normalizar si es necesario

    # Calcular sumas por enfermedad y sexo
    # Crear un DataFrame temporal para facilitar el uso con Plotly Express
    temp_df = pd.DataFrame({
        'disease': ['Dengue', 'Dengue', 'Zika', 'Zika', 'Chikungunya', 'Chikungunya'],
        'sex_category': ['Hombres', 'Mujeres', 'Hombres', 'Mujeres', 'Hombres', 'Mujeres'],
        'cases': [
            df[df['sex'] == 'M']['dengue_confirmed'].sum(),
            df[df['sex'] == 'F']['dengue_confirmed'].sum(),
            df[df['sex'] == 'M']['zika_confirmed'].sum(),
            df[df['sex'] == 'F']['zika_confirmed'].sum(),
            df[df['sex'] == 'M']['chik_confirmed'].sum(),
            df[df['sex'] == 'F']['chik_confirmed'].sum()
        ]
    })

    # Filtrar categorías con 0 casos para no mostrarlas en el pie chart
    temp_df = temp_df[temp_df['cases'] > 0]

    if temp_df.empty:
        print("No hay casos confirmados para graficar en el pie chart.")
        return

    # Crear una nueva columna para las etiquetas combinadas
    temp_df['label'] = temp_df['disease'] + ' - ' + temp_df['sex_category']

    # Crear el gráfico de torta con Plotly Express
    fig = px.pie(temp_df, 
                 values='cases', 
                 names='label', 
                 title='Proporción de Casos Confirmados por Enfermedad y Sexo',
                 color_discrete_sequence=px.colors.qualitative.Pastel, # Paleta de colores
                 hole=0.3, # Para un gráfico de "donut"
                 labels={'label': 'Categoría', 'cases': 'Número de Casos'}
                )

    fig.update_traces(textposition='inside', textinfo='percent+label', 
                      marker=dict(line=dict(color='#000000', width=1))) # Borde negro para las secciones

    fig.update_layout(
        title_font_size=18,
        title_x=0.5, # Centrar el título
        legend_title_text='Categorías de Casos'
    )
    
    fig.show()
    fig.write_html('disease_proportion_by_sex_pie.html')

    # Opcional: Imprimir las proporciones en consola para verificación
    total_cases = temp_df['cases'].sum()
    print("\nProporciones calculadas:")
    for index, row in temp_df.iterrows():
        percentage = (row['cases'] / total_cases) * 100
        print(f"{row['label']}: {row['cases']} casos ({percentage:.1f}%)")


def generate_visualizations_plotly(df_summary):
    """
    Genera visualizaciones clave de los factores de riesgo usando Plotly.
    """
    if df_summary is None:
        print("No hay datos para generar visualizaciones.")
        return

    # 1. Top 10 comunas por índice de riesgo
    top_10_risk = df_summary.head(10)
    fig_top_risk = px.bar(top_10_risk, x='comuna', y='risk_index', color='comuna',
                          title='Top 10 Comunas por Índice de Riesgo Combinado',
                          labels={'comuna': 'Comuna', 'risk_index': 'Índice de Riesgo', 'city': 'Ciudad'},
                          color_discrete_sequence=px.colors.qualitative.Plotly) 
    fig_top_risk.update_layout(xaxis_tickangle=-45)
    fig_top_risk.show()
    fig_top_risk.write_html("top_10_risk_comunas.html")

    # 2. Matriz de correlación
    correlation_matrix = df_summary[['normalized_total_cases', 'normalized_poverty_rate', 'normalized_no_services_rate', 'normalized_inverted_wolbachia', 'risk_index']].corr()
    fig_corr = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                         title='Matriz de Correlación de Factores de Riesgo',
                         color_continuous_scale=px.colors.sequential.RdBu) 
    fig_corr.show()
    fig_corr.write_html("correlation_matrix.html")

    # 3. Casos vs pobreza
    fig_poverty_cases = px.scatter(df_summary, x='avg_poverty_rate', y='total_cases', color='city', size='risk_index',
                                   title='Casos Totales vs. Tasa de Pobreza por Comuna',
                                   labels={'avg_poverty_rate': 'Tasa de Pobreza Promedio', 'total_cases': 'Total de Casos Confirmados', 'city': 'Ciudad', 'risk_index': 'Índice de Riesgo'},
                                   hover_name='comuna',
                                   color_discrete_sequence=px.colors.qualitative.D3) 
    fig_poverty_cases.show()
    fig_poverty_cases.write_html("cases_vs_poverty.html")

    # 4. Casos vs prevalencia Wolbachia
    fig_wolbachia_cases = px.scatter(df_summary, x='final_wolbachia_prevalence', y='total_cases', color='city', size='risk_index',
                                     title='Casos Totales vs. Prevalencia de Wolbachia por Comuna',
                                     labels={'final_wolbachia_prevalence': 'Prevalencia Final de Wolbachia', 'total_cases': 'Total de Casos Confirmados', 'city': 'Ciudad', 'risk_index': 'Índice de Riesgo'},
                                     hover_name='comuna',
                                     color_discrete_sequence=px.colors.qualitative.G10) 
    fig_wolbachia_cases.show()
    fig_wolbachia_cases.write_html("cases_vs_wolbachia.html")

def exploratory_data_analysis_plotly(df):
    """
    Realiza un EDA complementario y genera visualizaciones con Plotly:
    - Distribución de edades
    - Casos confirmados por sexo (ya cubierta por plot_cases_by_sex_plotly, pero se puede incluir aquí si se desea una versión diferente)
    - Evolución temporal de casos
    """
    print("\n--- Exploratory Data Analysis (EDA) ---\n")

    # 1. Valores faltantes
    print("Valores faltantes por columna:")
    print(df.isnull().sum())

    # 2. Estadísticas descriptivas
    print("\nEstadísticas descriptivas generales:")
    print(df.describe(include='all'))

    # 3. Distribución de edades con colores diferentes por bin
    if 'age' in df.columns:
        # Filtrar edades válidas (asumiendo que 'age' es numérica)
        ages = df['age'].dropna()
        if len(ages) > 0:
            # Crear el histograma con go.Histogram para control total
            fig_age_dist = go.Figure()
            
            # Calcular el histograma manualmente para asignar colores
            hist, bin_edges = np.histogram(ages, bins=30)  # 30 bins, ajusta si quieres más/menos
            
            # Crear las barras con go.Bar
            for i in range(len(hist)):
                color = px.colors.sequential.Inferno[i % len(px.colors.sequential.Inferno)]  
                
                fig_age_dist.add_trace(go.Bar(
                                x=[(bin_edges[i] + bin_edges[i+1]) / 2],  
                                y=[hist[i]],  
                                width=[bin_edges[i+1] - bin_edges[i]],  
                                marker_color=color,  
                                name=f'Bin {i+1} ({bin_edges[i]:.0f}-{bin_edges[i+1]:.0f})',  
                                showlegend=False  
                            ))
            
            # Configurar el layout
            fig_age_dist.update_layout(
                title='Distribución de edades de los casos (colores por rango de edad)',
                xaxis_title='Edad',
                yaxis_title='Frecuencia',
                bargap=0.1,  
                xaxis=dict(tickmode='linear', dtick=5) 
            )
            
            fig_age_dist.show()
            fig_age_dist.write_html("age_distribution.html")
            print("Gráfico de distribución de edades guardado como 'age_distribution.html' con colores por bin.")
        else:
            print("No hay datos válidos en la columna 'age'.")
    else:
        print("Columna 'age' no encontrada para distribución de edades.")
    
    # 4. Casos confirmados por sexo (ya cubierta por una función separada, pero se puede adaptar aquí si es necesario)
    if 'sex' in df.columns:
        sex_cases = df.groupby("sex")[["dengue_confirmed","zika_confirmed","chik_confirmed"]].sum()
        print("\nCasos confirmados por sexo:")
        print(sex_cases)
    else:
        print("Columna 'sex' no encontrada para casos por sexo.")

    # 5. Casos en el tiempo
    df['report_date'] = pd.to_datetime(df['report_date'])
    cases_time = df.groupby("report_date")[["dengue_confirmed","zika_confirmed","chik_confirmed"]].sum().reset_index()
    
    # Para una evolución temporal, es mejor usar melt para Plotly
    cases_time_melted = cases_time.melt(id_vars='report_date', var_name='disease', value_name='cases')
    
    # Aplicar una ventana de 30 días para suavizar la curva
    cases_time_melted['cases_rolling_30'] = cases_time_melted.groupby('disease')['cases'].transform(lambda x: x.rolling(30, min_periods=1).sum())

    fig_time_series = px.line(cases_time_melted, x='report_date', y='cases_rolling_30', color='disease',
                              title='Evolución de casos confirmados (ventana 30 días)',
                              labels={'report_date': 'Fecha de reporte', 'cases_rolling_30': 'Casos confirmados (ventana 30 días)', 'disease': 'Enfermedad'},
                              color_discrete_sequence=px.colors.qualitative.Pastel) 
    fig_time_series.show()
    fig_time_series.write_html("cases_time_series.html")

    print("\nEDA completado.\n")

if __name__ == "__main__":
    file_path = 'wolbachia_socio_epi_10k.csv'  # Asegúrate de que esta ruta sea correcta

    print("Cargando y preprocesando datos...")
    data = load_and_preprocess_data(file_path)

    if data is not None:
        print("\nDatos preprocesados (primeras 5 filas):")
        print(data.head())
        print("\nInformación de las columnas:")
        data.info()

        # Calcular factores de riesgo
        risk_summary = calculate_risk_factors(data)

        # Mostrar tabla de comunas con mayor riesgo combinado
        show_and_save_top_risk_table(risk_summary, top_n=10)

        # Graficar distribución de casos por enfermedad con Plotly
        plot_disease_distribution_plotly(data)

        # Graficar casos confirmados por sexo con Plotly
        plot_cases_by_sex_plotly(data)

        # ¡NUEVO! Graficar la proporción de casos por enfermedad y sexo en un pie chart con Plotly
        plot_disease_proportion_by_sex_pie_plotly(data)
        
        # Realizar análisis exploratorio de datos complementario con Plotly
        exploratory_data_analysis_plotly(data)

        print("\nGenerando visualizaciones adicionales con Plotly...")
        generate_visualizations_plotly(risk_summary)
        print("\nAnálisis completado. Las visualizaciones se han mostrado y guardado como HTML.")
    else:
        print("No se pudieron cargar los datos. Por favor, verifica la ruta del archivo.")  

    # --- CÁLCULO DE MÉTRICAS GLOBALES ---
    if data is not None:
        global_total_cases = data['total_confirmed_cases'].sum() 
        global_avg_wolbachia = data['wolbachia_prevalence'].mean() 
        global_avg_poverty = data['poverty_rate'].mean() 
        global_avg_no_services = data['no_services_rate'].mean() 
        print(f"\n--- Métricas Globales ---")
        print(f"Total Casos Confirmados: {global_total_cases:,.0f}")
        print(f"Prevalencia Wolbachia Promedio: {global_avg_wolbachia:.2%}")
        print(f"Tasa Pobreza Promedio: {global_avg_poverty:.2%}")
        print(f"Tasa No Servicios Promedio: {global_avg_no_services:.2%}")
    # --- FIN CÁLCULO DE MÉTRICAS GLOBALES ---