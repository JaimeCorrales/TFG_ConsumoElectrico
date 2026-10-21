FROM python:3.11-slim

# Directorio de trabajo estándar dentro del contenedor
WORKDIR /app

# Copiar e instalar dependencias 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código, modelos y archivos del proyecto
COPY . .

# Exponer el puerto por defecto de Streamlit
EXPOSE 8501

# Copiamos la carpeta de la aplicación dentro del contenedor
COPY aplicacion_final /app/aplicacion_final

# Comando indicando la ruta exacta al script
CMD ["streamlit", "run", "aplicacion_final/app.py", "--server.port=8501", "--server.address=0.0.0.0"]