# OCR
# Invoice OCR + Validator

Aplicación para procesar un lote de imágenes de facturas/recibos, extraer texto por OCR y validar montos (sumas / promedios). Soporta dos backends de OCR:

- `tesseract` (local, abierto)
- `google` (Google Cloud Vision API — mayor precisión en muchos casos)

Salida:
- CSV con campos extraídos por imagen
- Reporte con validación de sumas / discrepancias

## Requisitos
- Python 3.9+
- Tesseract (si usas backend `tesseract`) instalado en el sistema
- (opcional) Google Cloud service account JSON para Vision API

## Instalación (local)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
