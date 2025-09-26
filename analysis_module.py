import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

def plot_disease_distribution(df):
    """
    Grafica la distribución total de casos por enfermedad.
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

    plt.figure(figsize=(8,6))
    sns.barplot(x=diseases, y=cases, palette='viridis')
    plt.title('Distribución de Casos por Enfermedad')
    plt.ylabel('Número de Casos Confirmados')
    plt.xlabel('Enfermedad')
    plt.tight_layout()
    plt.savefig('disease_cases_distribution.png')  # Guarda la imagen
    plt.show()

def plot_cases_by_sex(df):
    """
    Grafica los casos confirmados por sexo para cada enfermedad.
    """
    if df is None or 'sex' not in df.columns:
        print("No hay datos o columna 'sex' para graficar casos por sexo.")
        return

    sex_cases = df.groupby('sex')[['dengue_confirmed', 'zika_confirmed', 'chik_confirmed']].sum().reset_index()

    # Transformar para gráfico tipo barras apiladas
    sex_cases_melted = sex_cases.melt(id_vars='sex', var_name='disease', value_name='cases')

    plt.figure(figsize=(10,6))
    sns.barplot(data=sex_cases_melted, x='sex', y='cases', hue='disease', palette='Set2')
    plt.title('Casos Confirmados por Sexo')
    plt.ylabel('Número de Casos Confirmados')
    plt.xlabel('Sexo')
    plt.tight_layout()
    plt.savefig('cases_by_sex.png')  # Guarda la imagen
    plt.show()

def generate_visualizations(df_summary):
    """
    Genera visualizaciones clave de los factores de riesgo.
    """
    if df_summary is None:
        print("No hay datos para generar visualizaciones.")
        return

    # Top 10 comunas por índice de riesgo
    top_10_risk = df_summary.head(10)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='comuna', y='risk_index', hue='city', data=top_10_risk)
    plt.title('Top 10 Comunas por Índice de Riesgo Combinado')
    plt.xlabel('Comuna')
    plt.ylabel('Índice de Riesgo')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Matriz de correlación
    correlation_matrix = df_summary[['normalized_total_cases', 'normalized_poverty_rate', 'normalized_no_services_rate', 'normalized_inverted_wolbachia', 'risk_index']].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlación de Factores de Riesgo')
    plt.tight_layout()
    plt.show()

    # Casos vs pobreza
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='avg_poverty_rate', y='total_cases', hue='city', size='risk_index', data=df_summary, sizes=(20, 400))
    plt.title('Casos Totales vs. Tasa de Pobreza por Comuna')
    plt.xlabel('Tasa de Pobreza Promedio')
    plt.ylabel('Total de Casos Confirmados')
    plt.tight_layout()
    plt.show()

    # Casos vs prevalencia Wolbachia
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='final_wolbachia_prevalence', y='total_cases', hue='city', size='risk_index', data=df_summary, sizes=(20, 400))
    plt.title('Casos Totales vs. Prevalencia de Wolbachia por Comuna')
    plt.xlabel('Prevalencia Final de Wolbachia')
    plt.ylabel('Total de Casos Confirmados')
    plt.tight_layout()
    plt.show()

def exploratory_data_analysis(df):
    """
    Realiza un EDA complementario:
    - Valores faltantes
    - Estadísticas descriptivas
    - Distribución de edades
    - Casos confirmados por sexo
    - Evolución temporal de casos
    """
    print("\n--- Exploratory Data Analysis (EDA) ---\n")

    # 1. Valores faltantes
    print("Valores faltantes por columna:")
    print(df.isnull().sum())

    # 2. Estadísticas descriptivas
    print("\nEstadísticas descriptivas generales:")
    print(df.describe(include='all'))

    # 3. Distribución de edades
    if 'age' in df.columns:
        plt.figure(figsize=(8,5))
        sns.histplot(df['age'], bins=30, kde=True, color="steelblue")
        plt.title("Distribución de edades de los casos")
        plt.xlabel("Edad")
        plt.ylabel("Frecuencia")
        plt.tight_layout()
        plt.show()
    else:
        print("Columna 'age' no encontrada para distribución de edades.")

    # 4. Casos confirmados por sexo
    if 'sex' in df.columns:
        sex_cases = df.groupby("sex")[["dengue_confirmed","zika_confirmed","chik_confirmed"]].sum()
        print("\nCasos confirmados por sexo:")
        print(sex_cases)
    else:
        print("Columna 'sex' no encontrada para casos por sexo.")

    # 5. Casos en el tiempo
    df['report_date'] = pd.to_datetime(df['report_date'])
    cases_time = df.groupby("report_date")[["dengue_confirmed","zika_confirmed","chik_confirmed"]].sum()

    plt.figure(figsize=(12,6))
    cases_time.rolling(30).sum().plot()
    plt.title("Evolución de casos confirmados (ventana 30 días)")
    plt.ylabel("Casos confirmados")
    plt.xlabel("Fecha de reporte")
    plt.tight_layout()
    plt.show()

    print("\nEDA completado.\n")

# NUEVO: Gráfico de torta para proporción de casos por enfermedad y sexo
def plot_disease_proportion_by_sex_pie(df):
    """
    Grafica la proporción de casos confirmados por enfermedad y sexo en un gráfico de torta.
    Desagrega por: Dengue/Zika/Chikungunya x Hombres/Mujeres.
    """
    if df is None:
        print("No hay datos para graficar la proporción de casos por enfermedad y sexo.")
        return

    # Asegurar que la columna 'sex' esté limpia (por ejemplo, 'M' para hombres, 'F' para mujeres)
    df['sex'] = df['sex'].str.upper().str.strip()  # Normalizar si es necesario

    # Calcular sumas por enfermedad y sexo
    # Dengue
    dengue_men = df[df['sex'] == 'M']['dengue_confirmed'].sum()
    dengue_women = df[df['sex'] == 'F']['dengue_confirmed'].sum()
    
    # Zika
    zika_men = df[df['sex'] == 'M']['zika_confirmed'].sum()
    zika_women = df[df['sex'] == 'F']['zika_confirmed'].sum()
    
    # Chikungunya
    chik_men = df[df['sex'] == 'M']['chik_confirmed'].sum()
    chik_women = df[df['sex'] == 'F']['chik_confirmed'].sum()

    # Lista de casos y labels
    cases = [dengue_men, dengue_women, zika_men, zika_women, chik_men, chik_women]
    labels = [
        'Dengue - Hombres',
        'Dengue - Mujeres',
        'Zika - Hombres',
        'Zika - Mujeres',
        'Chikungunya - Hombres',
        'Chikungunya - Mujeres'
    ]

    # Filtrar categorías con 0 casos para no mostrarlas en el pie chart
    filtered_cases = [c for c in cases if c > 0]
    filtered_labels = [label for label, c in zip(labels, cases) if c > 0]

    if not filtered_cases:
        print("No hay casos confirmados para graficar en el pie chart.")
        return

    # Configurar colores: alternando por sexo (azules para hombres, rosados para mujeres) para mayor claridad visual
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # Paleta variada de Matplotlib
    # O si prefieres diferenciar por sexo: 
    # colors_men = ['#1f77b4', '#2ca02c', '#9467bd']  # Azules/Verdes para hombres
    # colors_women = ['#ff7f0e', '#d62728', '#8c564b']  # Naranjas/Rojos para mujeres
    # colors = [colors_men[0], colors_women[0], colors_men[1], colors_women[1], colors_men[2], colors_women[2]]

    # Crear el gráfico
    plt.figure(figsize=(10, 10))  # Aumenté un poco el tamaño para mejor legibilidad con más slices
    wedges, texts, autotexts = plt.pie(
        filtered_cases, 
        labels=filtered_labels, 
        autopct='%1.1f%%',  # Muestra porcentajes con un decimal
        startangle=90,      # Inicia desde arriba para mejor distribución
        colors=colors[:len(filtered_cases)],  # Usa solo los colores necesarios
        explode=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05)[:len(filtered_cases)],  # Separa ligeramente cada slice para resaltar
        textprops={'fontsize': 10}  # Tamaño de fuente para labels
    )

    # Ajustar el tamaño de los porcentajes para que no se superpongan
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    plt.title('Proporción de Casos Confirmados por Enfermedad y Sexo\n(Distribución Relativa al Total de Casos)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.axis('equal')  # Asegura que sea un círculo perfecto
    plt.tight_layout()
    
    # Guardar la imagen
    plt.savefig('disease_proportion_by_sex_pie.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Opcional: Imprimir las proporciones en consola para verificación
    total_cases = sum(filtered_cases)
    print("\nProporciones calculadas:")
    for label, case in zip(filtered_labels, filtered_cases):
        percentage = (case / total_cases) * 100
        print(f"{label}: {case} casos ({percentage:.1f}%)")

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

        # Graficar distribución de casos por enfermedad
        plot_disease_distribution(data)

        # Graficar casos confirmados por sexo
        plot_cases_by_sex(data)

        # ¡NUEVO! Graficar la proporción de casos por enfermedad y sexo en un pie chart
        plot_disease_proportion_by_sex_pie(data)
        
        # Realizar análisis exploratorio de datos complementario
        exploratory_data_analysis(data)

        print("\nGenerando visualizaciones adicionales...")
        generate_visualizations(risk_summary)
        print("\nAnálisis completado. Las visualizaciones se han mostrado.")
    else:
        print("No se pudieron cargar los datos. Por favor, verifica la ruta del archivo.")  

 # --- CÁLCULO DE MÉTRICAS GLOBALES ---
    global_total_cases = data['total_confirmed_cases'].sum() # Suma de todos los casos en el dataset original
    global_avg_wolbachia = data['wolbachia_prevalence'].mean() # Promedio de la prevalencia original
    global_avg_poverty = data['poverty_rate'].mean() # Promedio de la tasa de pobreza original
    global_avg_no_services = data['no_services_rate'].mean() # Promedio de la tasa de no servicios original
    print(f"\n--- Métricas Globales ---")
    print(f"Total Casos Confirmados: {global_total_cases:,.0f}")
    print(f"Prevalencia Wolbachia Promedio: {global_avg_wolbachia:.2%}")
    print(f"Tasa Pobreza Promedio: {global_avg_poverty:.2%}")
    print(f"Tasa No Servicios Promedio: {global_avg_no_services:.2%}")
    # --- FIN CÁLCULO DE MÉTRICAS GLOBALES ---

#def export_and_insert_table(df_summary, html_file_path, top_n=10):
    """
    Genera la tabla HTML de las top N comunas con mayor riesgo y la inserta en el archivo HTML dado,
    reemplazando el contenido dentro del div con clase 'table-container'.
    """
#    if df_summary is None:
#        print("No hay datos para exportar e insertar.")
#        return

#    cols = ['city', 'comuna', 'total_cases', 'avg_poverty_rate', 'avg_no_services_rate', 'final_wolbachia_prevalence', 'risk_index']
#    top_risk = df_summary[cols].head(top_n)

    # Generar tabla HTML con clases para CSS y formato numérico
#    table_html = top_risk.to_html(index=False, classes='risk-table', float_format='%.3f')

    # Leer el archivo HTML original
#    with open(html_file_path, 'r', encoding='utf-8') as f:
#        html_content = f.read()

    # Patrón para encontrar el div con clase 'table-container'
#    pattern = re.compile(r'(<div class="table-container">)(.*?)(</div>)', re.DOTALL)

    # Reemplazar el contenido dentro del div por la tabla generada
#    new_html_content, count = pattern.subn(r'\1\n' + table_html + r'\n\3', html_content)

#    if count == 0:
#        print("No se encontró el div con clase 'table-container' en el archivo HTML.")
#        return

    # Guardar el archivo HTML modificado
#    with open(html_file_path, 'w', encoding='utf-8') as f:
#        f.write(new_html_content)

#    print(f"Tabla insertada correctamente en '{html_file_path}'.")

# --- Bloque principal ---

#if __name__ == "__main__":
#    file_path = 'wolbachia_socio_epi_10k.csv'  # Ruta a tu CSV

#    html_file_path = 'wolbachia.html'  # Ruta a tu archivo HTML donde insertar la tabla

#    print("Cargando y preprocesando datos...")
#    data = load_and_preprocess_data(file_path)

#    if data is not None:
#        print("Calculando factores de riesgo...")
#        risk_summary = calculate_risk_factors(data)

#        print("Mostrando tabla de comunas con mayor riesgo combinado:")
#        show_and_save_top_risk_table(risk_summary, top_n=10)

#        print("Insertando tabla en el archivo HTML...")
#        export_and_insert_table(risk_summary, html_file_path, top_n=10)

        # Aquí puedes seguir con tus otras funciones para gráficos, EDA, etc.
        # plot_disease_distribution(data)
        # plot_cases_by_sex(data)
        # exploratory_data_analysis(data)
        # generate_visualizations(risk_summary)

#        print("Proceso completado.")
#    else:
#        print("No se pudieron cargar los datos. Verifica la ruta del archivo.")

#import re

#def update_metrics_in_html_simple(html_path, output_path, total_cases, avg_wolbachia, avg_poverty, avg_no_services):
#    with open(html_path, 'r', encoding='utf-8') as f:
#        html_content = f.read()

    # Función para reemplazar el contenido entre <p id="some-id">...</p>
#    def replace_metric(html, element_id, new_value):
#        pattern = rf'(<p id="{element_id}">)(.*?)(</p>)'
#        replacement = rf'\1{new_value}\3'
#        return re.sub(pattern, replacement, html, flags=re.DOTALL)

    # Formatea los valores
#    total_cases_str = f"{total_cases:,}"
#    avg_wolbachia_str = f"{avg_wolbachia:.2f}%"
#    avg_poverty_str = f"{avg_poverty:.2f}%"
#    avg_no_services_str = f"{avg_no_services:.2f}%"

    # Reemplaza cada métrica
#    html_content = replace_metric(html_content, 'total-confirmed-cases', total_cases_str)
#    html_content = replace_metric(html_content, 'avg-wolbachia-prevalence', avg_wolbachia_str)
#    html_content = replace_metric(html_content, 'avg-poverty-rate', avg_poverty_str)
#    html_content = replace_metric(html_content, 'avg-no-services-rate', avg_no_services_str)

    # Guarda el archivo modificado
#    with open(output_path, 'w', encoding='utf-8') as f:
#        f.write(html_content)

#    print(f"Métricas actualizadas en {output_path}")

# Ejemplo de uso
#total_cases = 123456
#avg_wolbachia = 45.67
#avg_poverty = 23.45
#avg_no_services = 12.34

#html_file = 'wolbachia.html'
#output_file = 'wolbachia.html'

#update_metrics_in_html_simple(html_file, output_file, total_cases, avg_wolbachia, avg_poverty, avg_no_services)        