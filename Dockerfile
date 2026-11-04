FROM python:3.11-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo de requisitos e instalar dependencias primero
COPY aplicacion_final/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el contenido de aplicacion_final al directorio de trabajo
COPY aplicacion_final /app/aplicacion_final

# Exponer el puerto de Streamlit
EXPOSE 8501

# Comando de arranque (apuntando al app.py dentro de la carpeta)
CMD ["streamlit", "run", "aplicacion_final/app.py", "--server.port=8501", "--server.address=0.0.0.0"]