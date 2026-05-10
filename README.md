# TFM-brain-to-text-decoding

Repositorio del Trabajo Fin de Máster sobre decodificación de habla a partir de actividad neuronal mediante modelos "brain-to-text" fonémicos y textuales, rescoring con modelos de lenguaje y síntesis de voz.

## Descripción general

Este repositorio recoge el código, notebooks, configuraciones, scripts y resultados intermedios utilizados en el desarrollo experimental del trabajo. El proyecto se organiza en cuatro bloques principales:


- `EDA/`: análisis exploratorio de datos.

- `Error\_Analisys/`: análisis de errores de la rama fonémica y textual.

- `Experiments/`: experimentos principales del trabajo.

- `envs/`: entornos Conda y requisitos necesarios para reproducir las distintas partes del proyecto.

El trabajo parte del repositorio oficial del baseline:

- `https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text`

Ese repositorio proporciona la base original del sistema "brain-to-text", así como las instrucciones para descargar los datos, configurar el entorno baseline y montar la infraestructura de evaluación con modelo de lenguaje y Redis para la rama fonémica.

---

## Repositorio base de referencia

El modelo base utilizado en este trabajo proviene del repositorio oficial:

- `https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text`

Antes de ejecutar los experimentos de este repositorio, es necesario seguir las instrucciones del repositorio base para:



1\. Descargar los datos del benchmark original.

2\. Configurar el entorno del baseline.

3\. Preparar el modelo de lenguaje de la rama fonémica.

4\. Montar la evaluación con Redis, que en la rama fonémica de este trabajo sigue dependiendo de la infraestructura del baseline.



En este repositorio se incluyen adaptaciones, extensiones y nuevas arquitecturas, pero la preparación de datos y parte de la evaluación fonema-a-texto siguen apoyándose en la implementación original del baseline.



\---



\## Estructura del repositorio



```text

.

├── EDA/

├── Error\_Analisys/

├── Experiments/

├── envs/

├── LICENSE

├── README.md

└── data\_download\_link.txt



\---



\### `EDA/`



Contiene el análisis exploratorio de datos del proyecto.



Contenido principal:



* `EDA.ipynb`: notebook principal del análisis exploratorio.
* `img/`: figuras generadas durante el análisis exploratorio.



Figuras incluidas:



* distribución de secuencias de entrada,
* longitud de secuencias objetivo,
* número de ensayos por sesión,
* relación entre señal de entrada y objetivo,
* suavizado gaussiano,
* ejemplos de señal neural,
* variación temporal por característica.



\---



\### `Error\_Analisys/`

Contiene los notebooks de análisis de errores para ambas ramas del sistema.



\#### `Error\_Analisys/fonemas/`

* `Análisis\_Errores\_Fonémas.ipynb`: notebook de análisis de errores de la rama fonémica.
* `img\_fonemas/`: figuras generadas para el análisis fonémico.



Incluye análisis como:



* PER/WER por longitud de frase,
* PER/WER por sesión,
* matriz de confusión fonémica,
* relación entre tasa de error y frecuencia de fonema.



\#### `Error\_Analisys/texto/`

* `Análisis\_Errores\_TEXT.ipynb`: notebook de análisis de errores de la rama textual.
* `img\_text/`: figuras generadas para el análisis textual.



Incluye análisis como:



* CER/WER frente a longitud de frase,
* WER por sesión,
* matriz de confusión léxica,
* comparación entre greedy, beam y beam + LM.



\#### `Experiments/`

Contiene la parte principal de experimentación del trabajo y se divide en tres grandes bloques:



* `Phoneme/`
* `Text/`
* `TTS/`



