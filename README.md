# Destilación FDT

## Desarrollado por

- Jorwin Navarrete
- Erick Meléndez
- Edson Cruz

## Descripción del proyecto

**Destilación FDT** es el repositorio que alberga todo el material generado durante el curso de _Fenómenos de Transporte_ (quinto semestre de Ingeniería Química, Universidad Nacional de Ingeniería). El objetivo fue diseñar una columna de destilación multicomponente para recuperar gases ligeros del destilado de petróleo crudo.

## Prerrequisitos

Antes de ejecutar el proyecto en tu máquina local, necesitas contar con:

- [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
- [![Git](https://img.shields.io/badge/Git-%23F05033?style=for-the-badge&logo=git&logoColor=white)](https://git-scm.com/)

## Instalación

1. Clona el repositorio y accede a su carpeta:

   ```bash
   git clone https://github.com/jorwinnn/destilacion-fdt.git
   cd destilacion-fdt
   ```

2. Crea un entorno virtual e instala las dependencias:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # En Windows usa: .venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```

## Uso

Para abrir la interfaz gráfica que muestra los resultados de una simulación con entradas predeterminadas, ejecuta:

```bash
flet run -m ui.main
```

## Estructura del repositorio

- **paper/**
  Contiene el documento completo del proyecto: objetivos, introducción, justificación, marco teórico, desarrollo del proyecto, conclusiones y referencias del diseño de la columna de destilación.

- **modelo/**
  Incluye todo el código y los datos asociados a la simulación numérica de la columna multicomponente.

- **ui/**
  Ofrece una “calculadora” interactiva basada en Flet que permite visualizar, de forma gráfica, los resultados de la simulación numérica para las condiciones de entrada especificadas en el documento.
