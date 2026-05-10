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

---

## Estructura del repositorio

```text
.
├── EDA/
├── Error\_Analisys/
├── Experiments/
├── envs/
├── LICENSE
├── README.md
└── data\_download\_link.txt
```
---

### `EDA/`
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
---

### `Error\_Analisys/`
Contiene los notebooks de análisis de errores para ambas ramas del sistema.

#### `Error\_Analisys/fonemas/`

* `Análisis\_Errores\_Fonémas.ipynb`: notebook de análisis de errores de la rama fonémica.
* `img\_fonemas/`: figuras generadas para el análisis fonémico.

Incluye análisis como:

* PER/WER por longitud de frase,
* PER/WER por sesión,
* matriz de confusión fonémica,
* relación entre tasa de error y frecuencia de fonema.

#### `Error\_Analisys/texto/`

* `Análisis\_Errores\_TEXT.ipynb`: notebook de análisis de errores de la rama textual.
* `img\_text/`: figuras generadas para el análisis textual.

Incluye análisis como:

* CER/WER frente a longitud de frase,
* WER por sesión,
* matriz de confusión léxica,
* comparación entre greedy, beam y beam + LM.

#### `Experiments/`
Contiene la parte principal de experimentación del trabajo y se divide en tres grandes bloques:

* `Phoneme/`
* `Text/`
* `TTS/`

##### Experimentos de la rama fonémica
```text
Experiments/Phoneme/
├── baseline/
├── Mamba_Base/
└── ConvMambaGRU_Phoneme/
```
---
1. Experiments/Phoneme/baseline/

Implementación y adaptación local del modelo baseline fonémico.

Archivos principales
* README.md: notas específicas del baseline local.
* LINK_REPO_BASELINE.txt: enlace al repositorio original del baseline.
* setup.sh: script auxiliar de preparación del entorno baseline.
* setup_lm.sh: script auxiliar para preparar el modelo de lenguaje del baseline.
* data_augmentations.py: transformaciones y aumentos de datos aplicados a la señal neural durante entrenamiento.
* dataset.py: definición del dataset y del cargador de datos para el baseline fonémico.
* rnn_model.py: implementación del modelo GRU baseline fonémico.
* rnn_trainer.py: lógica de entrenamiento del modelo baseline.
* train_model.py: script principal de entrenamiento del baseline fonémico.
* evaluate_model.py: evaluación del baseline fonémico sobre el conjunto de validación/test.
* evaluate_model_helpers.py: utilidades de evaluación compartidas.
* rnn_args.yaml: configuración de entrenamiento y modelo.
* nejm_b2txt_utils/: utilidades auxiliares derivadas de la estructura del baseline original.
* results/: resultados de validación en formato CSV.
* trained_models/: estructura local de modelos entrenados y logs.

##### Notas importantes

La rama fonémica baseline depende del modelo de lenguaje y del pipeline con Redis del repositorio original. Para reproducir la evaluación fonema-a-texto es necesario:

1. Seguir las instrucciones del repositorio original.
2. Preparar el LM del baseline.
3. Levantar Redis.
4. Ejecutar la evaluación usando la infraestructura del baseline.
