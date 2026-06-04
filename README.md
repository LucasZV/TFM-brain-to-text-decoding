# TFM-brain-to-text-decoding

Repositorio del Trabajo Fin de Máster sobre decodificación de habla a partir de actividad neuronal mediante modelos "brain-to-text" fonémicos y textuales, rescoring con modelos de lenguaje y síntesis de voz.

## Descripción general

Este repositorio recoge el código, notebooks, configuraciones, scripts y resultados intermedios utilizados en el desarrollo experimental del trabajo.  
El proyecto se organiza en cuatro bloques principales:


- `EDA/`: análisis exploratorio de datos.

- `Error_Analisys/`: análisis de errores de la rama fonémica y textual.

- `Experiments/`: experimentos principales del trabajo.

- `envs/`: entornos Conda y requisitos necesarios para reproducir las distintas partes del proyecto.

El trabajo parte del repositorio oficial del baseline:

- `https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text`

Ese repositorio proporciona la base original del sistema "brain-to-text", así como las instrucciones para descargar los datos, configurar el entorno baseline y montar la infraestructura de evaluación con modelo de lenguaje y Redis para la rama fonémica.  

Se pueden escuchar ejemplos del audio final sintetizado en la siguiente página desplegada:  
- https://lucaszv.github.io/TFM-audio-examples/ 

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
├── Error_Analisys/
├── Experiments/
├── envs/
├── LICENSE
├── README.md
└── data_download_link.txt
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

### `Error_Analisys/`
Contiene los notebooks de análisis de errores para ambas ramas del sistema.

#### `Error_Analisys/fonemas/`

* `Análisis_Errores_Fonémas.ipynb`: notebook de análisis de errores de la rama fonémica.
* `img_fonemas/`: figuras generadas para el análisis fonémico.

Incluye análisis como:

* PER/WER por longitud de frase,
* PER/WER por sesión,
* matriz de confusión fonémica,
* relación entre tasa de error y frecuencia de fonema.

#### `Error_Analisys/texto/`

* `Análisis_Errores_TEXT.ipynb`: notebook de análisis de errores de la rama textual.
* `img_text/`: figuras generadas para el análisis textual.

Incluye análisis como:

* CER/WER frente a longitud de frase,
* WER por sesión,
* matriz de confusión léxica,
* comparación entre greedy, beam y beam + LM.

## `Experiments/`
Contiene la parte principal de experimentación del trabajo y se divide en tres grandes bloques:

* `Phoneme/`
* `Text/`
* `TTS/`

### Experimentos de la rama fonémica
```text
Experiments/Phoneme/
├── baseline/
├── Mamba_Base/
└── ConvMambaGRU_Phoneme/
```
---
#### 1. `Experiments/Phoneme/baseline/`

Implementación y adaptación local del modelo baseline fonémico.

Archivos principales
* `README.md`: notas específicas del baseline local.
* `LINK_REPO_BASELINE.txt`: enlace al repositorio original del baseline.
* `setup.sh`: script auxiliar de preparación del entorno baseline.
* `setup_lm.sh`: script auxiliar para preparar el modelo de lenguaje del baseline.
* `data_augmentations.py`: transformaciones y aumentos de datos aplicados a la señal neural durante entrenamiento.
* `dataset.py`: definición del dataset y del cargador de datos para el baseline fonémico.
* `rnn_model.py`: implementación del modelo GRU baseline fonémico.
* `rnn_trainer.py`: lógica de entrenamiento del modelo baseline.
* `train_model.py`: script principal de entrenamiento del baseline fonémico.
* `evaluate_model.py`: evaluación del baseline fonémico sobre el conjunto de validación/test.
* `evaluate_model_helpers.py`: utilidades de evaluación compartidas.
* `rnn_args.yaml`: configuración de entrenamiento y modelo.
* `nejm_b2txt_utils/`: utilidades auxiliares derivadas de la estructura del baseline original.
* `results/`: resultados de validación en formato CSV.
* `trained_models/`: estructura local de modelos entrenados y logs.

##### Notas importantes

La rama fonémica baseline depende del modelo de lenguaje y del pipeline con Redis del repositorio original. Para reproducir la evaluación fonema-a-texto es necesario:

1. Seguir las instrucciones del repositorio original.
2. Preparar el LM del baseline.
3. Levantar Redis.
4. Ejecutar la evaluación usando la infraestructura del baseline.
---

#### 2. `Experiments/Phoneme/Mamba_Base/`
Implementación de la variante base basada en Mamba para la rama fonémica.

Archivos principales
* `data_augmentations.py`: aumentos de datos para la rama Mamba fonémica.
* `dataset.py`: definición del dataset para esta variante.
* `mamba_model.py`: implementación del modelo base Mamba fonémico.
* `mamba_trainer.py`: lógica de entrenamiento del modelo Mamba.
* `train_model.py`: script principal de entrenamiento.
* `evaluate_mamba_model.py`: evaluación del modelo Mamba fonémico.
* `evaluate_model_helpers.py`: utilidades compartidas para evaluación.
* `mamba_args.yaml`: configuración del modelo y entrenamiento.
* `test_mamba_forward.py`: test básico del paso forward del modelo.
* `test_mamba_ctc.py`: test básico de compatibilidad con pérdida CTC.
* `trained_models/`: resultados, logs y métricas del entrenamiento.

Contenido de `trained_models/`
* `baseline_mamba_nocompile/`: directorio de entrenamiento concreto.
* `checkpoint/args.yaml`: configuración efectiva.
* `checkpoint/val_metrics.pkl`: métricas de validación guardadas.
* `baseline_mamba_val_predicted_sentences_*.csv`: predicciones textuales validadas tras LM.
* `train_val_trials.json`: partición entrenamiento/validación.
* `training_log`: log del entrenamiento.

---
#### 3. `Experiments/Phoneme/ConvMambaGRU_Phoneme/`
Implementación de la arquitectura híbrida ConvMambaGRU para la rama fonémica.

Archivos principales
* `ConvMamba_model.py`: definición del núcleo híbrido Conv + Mamba + GRU.
* `convmamba_phoneme_trainer.py`: clase de entrenamiento de esta arquitectura.
* `train_convmamba_phoneme_model.py`: script principal de entrenamiento.
* `convmamba_phoneme_args.yaml`: configuración de entrenamiento/modelo.
* `data_augmentations.py`: aumentos de datos aplicados a esta rama.
* `phoneme_utils.py`: utilidades específicas de representación y decodificación fonémica.
* `evaluate_convmamba_phoneme_to_text.py`: evaluación fonema-a-texto.
* `evaluate_convmamba_rescoring.py`: evaluación del rescoring sobre salidas fonémicas.
* `evaluate_mamba_with_redis.py`: evaluación integrada con Redis/LM del baseline.
* `evaluate_model_helpers.py`: utilidades de evaluación compartidas.
* `rescoring_results/`: mejores resultados de rescoring en formato JSON.
* `results_val/`: salidas de validación en CSV.
* `trained_models/`: modelos entrenados, configuración y logs.
  
##### Notas importantes:

Al igual que el baseline fonémico, esta rama utiliza evaluación apoyada en Redis y en el modelo de lenguaje del repositorio base. Por tanto, para reproducir la evaluación final de WER de la rama fonémica hay que seguir las instrucciones del repositorio original y disponer del LM del baseline funcionando.

---
### Experimentos de la rama textual
```text
Experiments/Text/
├── GRU_Text/
└── ConvMambaGRU_Text/
```
---
#### 1. `Experiments/Text/GRU_Text/`
Implementación del modelo GRU textual para decodificación directa a caracteres.

Archivos principales
* `data_augmentations.py`: aumentos de datos aplicados a la señal neural.
* `text_dataset.py`: dataset y cargador de datos para la rama textual.
* `gru_text_model.py`: implementación del modelo GRU textual.
* `gru_text_trainer.py`: lógica de entrenamiento del modelo.
* `train_gru_text_model.py`: script principal de entrenamiento.
* `evaluate_gru_text_rescoring.py`: evaluación y rescoring del modelo textual.
* `evaluate_model_helpers.py`: utilidades compartidas de evaluación.
* `text_utils.py`: utilidades de vocabulario, decodificación CTC y procesamiento textual.
* `gru_text_args.yaml`: configuración de modelo y entrenamiento.
* `rescoring_results/`: barridos de alpha y resultados de rescoring.
* `trained_models/`: resultados del entrenamiento, configuración y métricas.
Contenido relevante
* `rescoring_results/alpha_sweep_gpt2xl_bw100/`: barrido de alpha para GPT-2 XL.
* `trained_models/gru_text/checkpoint/args.yaml`: configuración efectiva.
* `trained_models/gru_text/checkpoint/val_metrics.pkl`: métricas de validación.

#### 2. `Experiments/Text/ConvMambaGRU_Text/`
Implementación de la arquitectura ConvMambaGRU para decodificación textual directa.

Archivos principales
* `ConvMamba_model.py`: definición del bloque híbrido Mamba.
* `ConvMambaGru_text_model.py`: implementación completa del modelo textual híbrido.
* `convmamba_text_trainer.py`: lógica de entrenamiento.
* `train_convmamba_text_model.py`: script principal de entrenamiento.
* `convmamba_text_args.yaml`: configuración de entrenamiento/modelo.
* `data_augmentations.py`: aumentos de datos de esta rama.
* `text_dataset.py`: dataset textual.
* `text_utils.py`: utilidades de texto y CTC.
* `evaluate_convmamba_text_rescoring.py`: evaluación y rescoring del modelo textual.
* `evaluate_model_helpers.py`: utilidades comunes de evaluación.
* `rescoring_results/`: resultados de rescoring, barridos de alpha y resúmenes.
* `trained_models/`: modelos entrenados y métricas.
* `KenLM/`: utilidades para construir, entrenar y puntuar modelos KenLM sobre el corpus textual.

##### Carpeta KenLM/
Incluye herramientas auxiliares para crear y evaluar modelos n-gram específicos del corpus textual del trabajo:

* `build_corpus.py`: construcción del corpus textual.
* `train_kenlm.py`: entrenamiento de modelos KenLM.
* `train_kenlm.sh`: script auxiliar de entrenamiento.
* `score_kenlm.py`: puntuación de hipótesis con KenLM.
* `data/`: estadísticas y vocabulario.
* `corpus/`: corpus de entrenamiento/validación.
* `models/`: modelos ARPA y binarios generados.
##### *Nota sobre `kenlm_src/`*

La carpeta kenlm_src/ contiene el código fuente de KenLM y su build local. Es una dependencia auxiliar utilizada durante los experimentos de rescoring con n-gramas.

--- 

## Síntesis de voz
```text
Experiments/TTS/
├── tts_from_gru_baseline.py
├── tts_from_gru_phoneme_best.py
├── tts_from_gru_text_best.py
└── tts_outputs/
```
La carpeta `TTS/` contiene la etapa final de síntesis de voz a partir de frases decodificadas.

### Scripts principales
* `tts_from_gru_baseline.py`: genera audio a partir de las predicciones del modelo baseline fonémico.
* `tts_from_gru_phoneme_best.py`: genera audio a partir del mejor modelo de la rama fonémica seleccionada.
* `tts_from_gru_text_best.py`: genera audio a partir del mejor modelo de la rama textual.
### Salidas
* `tts_outputs/gru_baseline_csv_best/`: ejemplos sintetizados a partir del baseline.
* `tts_outputs/convmamba_phoneme_csv_best/`: ejemplos sintetizados a partir de la mejor rama fonémica.
* `tts_outputs/gru_text_best_gpt2xl_a12/`: ejemplos sintetizados a partir del mejor modelo textual.

Cada ejemplo incluye:

* un archivo `.txt` con la frase utilizada,
* un archivo `.wav` con el audio sintetizado.
---

## Entornos y dependencias

La carpeta `envs/` contiene los entornos y requisitos necesarios para reproducir el proyecto:

* `environment_b2txt25.yml`
* `environment_b2txt25_lm.yml`
* `environment_b2tmamba.yml`
* `environment_coqui_tts.yml`
* `requirements_b2tmamba.txt`
* `requirements_b2txt25.txt`
* `requirements_b2txt25_lm.txt`
* `requirements_coqui_tts.txt`

### Entorno general de experimentación:
```bash
conda env create -f envs/environment_b2tmamba.yml
```

### Entorno de rescoring / LM rama fonética:
```bash
conda env create -f envs/environment_b2txt25_lm.yml
```

### Entorno de síntesis de voz:
```bash
conda env create -f envs/environment_coqui_tts.yml
```
---

## Flujo recomendado de reproducción
1. Preparar el baseline original

Seguir las instrucciones del repositorio: https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text para:

* descargar los datos,
* montar el entorno original,
* preparar el modelo de lenguaje baseline,
* y configurar Redis para la rama fonémica.
2. Crear los entornos locales

Crear los entornos Conda de este repositorio desde la carpeta `envs/`.

3. Ejecutar el análisis exploratorio

Abrir y ejecutar:
* `EDA/EDA.ipynb`

4. Ejecutar la rama fonémica

Trabajar dentro de:

* `Experiments/Phoneme/baseline/`
* `Experiments/Phoneme/Mamba_Base/`
* `Experiments/Phoneme/ConvMambaGRU_Phoneme/`  

Orden orientativo  

* entrenar el baseline fonémico,
* entrenar la variante Mamba base,
* entrenar ConvMambaGRU fonémico,
* evaluar con Redis + LM del baseline,
* comparar resultados en PER/WER.
5. Ejecutar la rama textual

Trabajar dentro de:

* `Experiments/Text/GRU_Text/`
* `Experiments/Text/ConvMambaGRU_Text/`
Orden orientativo
* entrenar el modelo GRU textual,
* entrenar ConvMambaGRU textual,
* ejecutar rescoring,
* comparar CER/WER.
6. Ejecutar análisis de errores

Abrir los notebooks:

* `Error_Analisys/fonemas/Análisis_Errores_Fonémas.ipynb`
* `Error_Analisys/texto/Análisis_Errores_TEXT.ipynb`
7. Generar síntesis de voz

Trabajar dentro de:

* `Experiments/TTS/` usando el entorno `coqui_tts`.
--- 
### Archivos grandes no incluidos
Por restricciones de tamaño, este repositorio no incluye:

* checkpoints finales completos de los modelos entrenados,
* algunos logs pesados de entrenamiento,
* ciertos artefactos grandes generados durante evaluación y rescoring.

Sí se incluyen:

* scripts,
* configuraciones,
* notebooks,
* métricas intermedias,
* resultados resumidos,
* ejemplos de salidas,
* y la estructura necesaria para reproducir el pipeline.
--- 
## Descarga de datos

En la raíz del repositorio se incluye:

* `data_download_link.txt`

como referencia adicional para la descarga o localización de los datos empleados.

Aun así, la fuente principal y las instrucciones de preparación deben tomarse del repositorio base del baseline.

## Licencia

Este repositorio se distribuye bajo la licencia indicada en el archivo `LICENSE`.

Salvo indicación expresa en contrario, la memoria, figuras, tablas y materiales originales del trabajo siguen la misma licencia especificada para la documentación del proyecto.

### Observaciones
* La evaluación final de la rama fonémica depende del pipeline del baseline y del uso de Redis con el LM del repositorio original.
* La rama textual es más autocontenida dentro de este repositorio, aunque depende igualmente de los datos preparados a partir del baseline.
* La parte de TTS se ejecuta en un entorno separado para evitar conflictos de dependencias con los entornos de entrenamiento y evaluación.
