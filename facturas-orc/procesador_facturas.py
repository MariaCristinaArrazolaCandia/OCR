# procesador_facturas.py

import os
import re
import cv2
import pytesseract
import pandas as pd
from decimal import Decimal, InvalidOperation

# --- CONFIGURACIÓN ---
# Si Tesseract no está en el PATH de tu sistema, descomenta y ajusta la siguiente línea
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configura la variable de entorno para tessdata
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# Rutas de las carpetas (basado en la estructura del proyecto)
RUTA_IMAGENES = 'facturas_a_procesar'
RUTA_REPORTES = 'reportes'

# --- PASO 4: PRE-PROCESAMIENTO DE IMAGEN ---
def preprocesar_imagen(ruta_imagen):
    """
    Carga una imagen y la pre-procesa para mejorar la calidad del OCR.
    - Convierte a escala de grises.
    - Aplica un umbral para binarizar la imagen (blanco y negro puro).
    """
    img = cv2.imread(ruta_imagen)
    if img is None:
        return None
    
    # Convertir a escala de grises
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbral adaptativo para manejar diferentes iluminaciones
    # Esto es generalmente mejor que un umbral simple para facturas.
    binaria = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)

    return binaria

# --- Funciones de ayuda para normalización ---
def normalizar_monto(monto_str):
    """
    Convierte un string de monto (con posibles comas y puntos) a un objeto Decimal.
    Ej: "1.234,56" -> Decimal('1234.56'), "78.90" -> Decimal('78.90')
    """
    # Reemplaza coma por punto para el decimal y elimina puntos de miles
    monto_str_normalizado = monto_str.replace('.', '').replace(',', '.')
    partes = monto_str_normalizado.split('.')
    if len(partes) > 2:
        monto_str_normalizado = "".join(partes[:-1]) + "." + partes[-1]
    try:
        return Decimal(monto_str_normalizado)
    except (InvalidOperation, TypeError):
        return None

# --- NUEVO: RECONOCIMIENTO ESTRUCTURADO DE FACTURA ---
def reconocer_factura(imagen_preprocesada):
    """
    Usa pytesseract.image_to_data para obtener texto con posiciones y reconstruir la estructura de la factura,
    similar a un análisis manual: identificar encabezado, tabla de detalles y total.
    """
    # Configuración para Tesseract
    config = r'--oem 3 --psm 6 -l spa'
    
    # Obtener datos con posiciones
    data = pytesseract.image_to_data(imagen_preprocesada, config=config, output_type=pytesseract.Output.DICT)

    # Extraer palabras con posiciones (bounding boxes)
    palabras = []
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if text and int(data['conf'][i]) > 60:  # Filtrar por confianza > 60%
            palabras.append({
                'text': text,
                'left': data['left'][i],
                'top': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i]
            })

    # Ordenar palabras por posición: primero por fila (top), luego por columna (left)
    palabras.sort(key=lambda x: (x['top'], x['left']))

    # Agrupar palabras en líneas basadas en proximidad vertical
    lineas = []
    if palabras:
        linea_actual = [palabras[0]]
        for i in range(1, len(palabras)):
            palabra_anterior = linea_actual[-1]
            palabra_actual = palabras[i]
            # Si la palabra está en la misma línea (verticalmente cercana)
            if abs(palabra_actual['top'] - palabra_anterior['top']) < 20:
                linea_actual.append(palabra_actual)
            else:
                lineas.append(sorted(linea_actual, key=lambda p: p['left']))
                linea_actual = [palabra_actual]
        lineas.append(sorted(linea_actual, key=lambda p: p['left']))

    # Ahora, parsear las líneas para extraer estructura
    detalles = []
    total_factura = None
    columnas = {}
    estado = 'buscando_columnas'  # Estados: buscando_columnas, extrayendo_detalles, buscando_total

    # Definir posibles nombres para cada columna
    MAPEO_COLUMNAS = {
        'Cant': ['cant', 'cantidad', 'qty'],
        'Descripción': ['descripción', 'descripcion', 'producto', 'servicio', 'concepto'],
        'P.Unit': ['p.unit', 'unitario', 'precio', 'p/u'],
        'Importe': ['importe', 'total', 'subtotal', 'valor']
    }

    for linea in lineas:
        linea_texto = ' '.join(p['text'] for p in linea).lower()

        # --- Detección de Columnas (Estado: buscando_columnas) ---
        if estado == 'buscando_columnas':
            found_cols = False
            for nombre_col, alias_list in MAPEO_COLUMNAS.items():
                for alias in alias_list:
                    if alias in linea_texto:
                        # Encontrar la palabra exacta y su posición
                        for palabra in linea:
                            if alias in palabra['text'].lower():
                                columnas[nombre_col] = palabra['left']
                                found_cols = True
                                break
            if found_cols:
                # Si encontramos 'importe' en la línea de encabezados, no lo confundamos con el total final
                if 'Importe' in columnas and 'total' in linea_texto:
                    pass # Es un encabezado de columna, no el total final

                print(f"Columnas detectadas y sus posiciones 'left': {columnas}")
                estado = 'extrayendo_detalles'
                continue

        # --- Detección de la línea de Total ---
        if 'total' in linea_texto and estado != 'buscando_columnas':
            # Extraer el monto de la línea del total
            montos_en_linea = [normalizar_monto(p['text']) for p in linea if normalizar_monto(p['text']) is not None]
            if montos_en_linea:
                total_factura = montos_en_linea[-1] # El total suele ser el último monto en la línea
                estado = 'buscando_total' # Dejamos de buscar detalles
                continue

        # --- Extracción de Detalles (Estado: extrayendo_detalles) ---
        if estado == 'extrayendo_detalles' and columnas:
            # Ignorar líneas que parecen ser encabezados residuales
            if any(alias in linea_texto for col_aliases in MAPEO_COLUMNAS.values() for alias in col_aliases):
                continue

            # Asignar cada palabra de la línea a una columna por proximidad horizontal
            detalle_linea = {nombre_col: [] for nombre_col in columnas}
            for palabra in linea:
                # Encontrar la columna más cercana a la palabra
                distancias = {nombre_col: abs(palabra['left'] - pos_col) for nombre_col, pos_col in columnas.items()}
                columna_cercana = min(distancias, key=distancias.get)
                detalle_linea[columna_cercana].append(palabra['text'])

            # Consolidar y convertir los datos del detalle
            try:
                # Unir textos de descripción
                desc = ' '.join(detalle_linea.get('Descripción', []))
                # Tomar el primer valor para cantidad y precios
                cant_str = detalle_linea.get('Cant', [None])[0]
                punit_str = detalle_linea.get('P.Unit', [None])[0]
                importe_str = detalle_linea.get('Importe', [None])[0]

                # Solo procesar si tenemos un importe
                if importe_str and desc:
                    importe = normalizar_monto(importe_str)
                    if importe is not None:
                        detalles.append({
                            'Cant': int(float(cant_str.replace(',', '.'))) if cant_str else 1,
                            'Descripción': desc,
                            'P.Unit': normalizar_monto(punit_str) if punit_str else importe,
                            'Importe': importe
                        })
            except (ValueError, InvalidOperation, IndexError):
                pass # Ignorar líneas que no se pueden parsear

    # Si no se encontraron columnas, no se pueden extraer detalles
    if not columnas:
            estado = 'extrayendo_detalles'

    # Calcular total de detalles
    total_calculado = sum(d['Importe'] for d in detalles) if detalles else Decimal('0')
    
    return detalles, total_factura, total_calculado

# --- FUNCIÓN PRINCIPAL ---
def procesar_lote_facturas():
    """
    Función principal que orquesta todo el proceso.
    """
    print("Iniciando procesamiento de facturas...")
    
    if not os.path.exists(RUTA_IMAGENES):
        print(f"Error: La carpeta '{RUTA_IMAGENES}' no existe. Por favor, créala y añade imágenes.")
        return

    if not os.path.exists(RUTA_REPORTES):
        os.makedirs(RUTA_REPORTES)

    resultados = []
    archivos_imagen = [f for f in os.listdir(RUTA_IMAGENES) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]

    # --- PASO 3: PROCESAMIENTO POR LOTES ---
    for nombre_archivo in archivos_imagen:
        ruta_completa = os.path.join(RUTA_IMAGENES, nombre_archivo)
        print(f"\n Archivo: {nombre_archivo}")

        # 1. Pre-procesar la imagen
        imagen_preprocesada = preprocesar_imagen(ruta_completa)
        if imagen_preprocesada is None:
            print(f"No se pudo leer la imagen: {nombre_archivo}")
            resultados.append({
                'Archivo': nombre_archivo, 'Total Calculado': 'N/A', 'Total Factura': 'N/A',
                'Coherente': 'No', 'Error': 'No se pudo leer la imagen'
            })
            continue

        # Nuevo reconocimiento estructurado
        detalles, total_factura, total_calculado = reconocer_factura(imagen_preprocesada)
        
        if not detalles or total_factura is None:
            print("No se pudieron extraer suficientes datos.")
            resultados.append({
                'Archivo': nombre_archivo, 'Total Calculado': 'N/A', 'Total Factura': 'N/A',
                'Coherente': 'No', 'Error': 'No se extrajeron detalles o total'
            })
            continue

        # --- PASO 7: VALIDACIÓN DE LA LÓGICA ---
        es_coherente = (total_calculado == total_factura)

        print("Detalles encontrados:")
        if detalles:
            df_detalles = pd.DataFrame(detalles)
            print(df_detalles.to_string(index=False))
        else:
            print("No se encontraron detalles.")
        print(f"\nTotal Calculado (Suma de detalles): {total_calculado:.2f}")
        print(f"Total encontrado en Factura: {total_factura:.2f}")
        print(f"¿Es coherente?: {'Sí' if es_coherente else 'No'}")

        resultados.append({
            'Archivo': nombre_archivo,
            'Total Calculado': f"{total_calculado:.2f}",
            'Total Factura': f"{total_factura:.2f}",
            'Coherente': 'Sí' if es_coherente else 'No',
            'Error': ''
        })

    # --- PASO 8: GENERACIÓN DE UN REPORTE ---
    if resultados:
        ruta_txt = os.path.join(RUTA_REPORTES, 'reporte_facturas.txt')
        with open(ruta_txt, 'w', encoding='utf-8') as f:
            for resultado in resultados:
                f.write("========================================\n")
                f.write(f"Archivo: {resultado['Archivo']}\n")
                f.write("----------------------------------------\n")
                f.write(f"Total Calculado: {resultado['Total Calculado']}\n")
                f.write(f"Total Factura: {resultado['Total Factura']}\n")
                f.write(f"Coherente: {resultado['Coherente']}\n")
                if resultado['Error']:
                    f.write(f"Error: {resultado['Error']}\n")
                f.write("========================================\n\n")
        print(f"\nProceso completado. Reporte guardado en: {ruta_txt}")
    else:
        print("\nNo se encontraron imágenes para procesar.")

if __name__ == '__main__':
    procesar_lote_facturas()