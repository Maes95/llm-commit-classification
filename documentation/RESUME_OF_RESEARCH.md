# Evaluación de Modelos Grandes de Lenguaje para Anotación de Commits

## 1. Contexto y objetivo

El objetivo de esta investigación es evaluar si los modelos grandes de lenguaje (LLMs) pueden anotar commits de manera similar a un humano.

Como trabajo previo, se cuenta con una anotación manual de 1000 commits del kernel de Linux, realizada por tres anotadores humanos (A, B, C). Cada commit fue anotado con cuatro dimensiones asignando una puntuación de 0 a 4 (0 = no, 4 = sí, con grados intermedios para casos ambiguos, permitiendo anotar un mismo commit en más de una categoría):
- **BFC** (Bug Fixing Commit): ¿El commit corrige un bug? (0-4)
- **BPC** (Bug Preventing Commit): ¿El commit previene un bug? (0-4)
- **PRC** (Perfective Commit): ¿El commit mejora la calidad sin corregir o prevenir un bug? (0-4)
- **NFC** (New Functionality Commit): ¿El commit introduce nueva funcionalidad? (0-4)

El proceso de experimentación se divide en dos fases: Experimentación y Análisis.

## 2. Fase de experimentación

Se realizó una anotación de un subconjunto de 50 commits utilizando varios LLMs y configuraciones de contexto. 

### 2.1 LLMs evaluados:
- `gpt-5-mini`
- CodeLlama variantes: `codellama_34b`, `codellama_70b`
- DeepSeek variantes: `deepseek-coder-33b`, `deepseek-r1-32b`, `deepseek-r1-70b`
- `gpt-oss` variantes: `gpt-oss_20b`, `gpt-oss_120b`
- `llama4_16x17b`
- `qwen3-coder-30b`
- `gemma4-31b`

### 2.2 Configuraciones de contexto (rounds):
Sobre cada modelo, se realizaron anotaciones bajo diferentes configuraciones de contexto para evaluar su impacto en la calidad de las anotaciones:
- `r1`: Solo mensaje del commit (`message`)
- `r2`: Etiqueta de una sola dimensión (`single-label`)
- `r3`: `single-label` + ejemplos (`few-shot`)
- `r4`: `diff` + `single-label`
- `r5`: `diff` + `single-label` + `few-shot`

En el caso de `few-shot`, se proporcionaron ejemplos de commits anotados por humanos para guiar al modelo. Estos ejemplos se seleccionaron cuidadosamente para cubrir casos representativos de un acuerdo total, un acuerdo parcial y un desacuerdo total entre los anotadores humanos, con el objetivo de mejorar la capacidad del modelo para manejar casos ambiguos. Pueden encontrarse en `documentation/few-shot-examples.md`.


### 2.3 Proceso de anotación

La anotación se llevó a cabo utilizando el script `annotate_validation_set.py`, que orquesta la lectura del conjunto de validación, la generación de prompts, la invocación de los modelos y la persistencia de los resultados en la carpeta `output/`. 

#### 2.3.1 Configuración de los modelos

Cada modelo se ejecutó con `temperature=0.0` para maximizar la determinismo, y `max_tokens` se estableció en 10000 para reducir la incidencia de respuestas JSON truncadas, especialmente en modelos más verbosos. 

### 2.3.2 Diseño del prompt

En el prompt se incluyó:
- El contexto ofrecido a los anotadores humanos para realizar su tarea, extraído de `documentation/context.md`.
- Las definiciones (taxonomía) de cada dimensión (BFC, BPC, PRC, NFC) ofrecida a los humanos, obtenidas del fichero `documentation/definitions.md`
- Instrucciones especificas sobre la tarea de anotación, incluyendo la escala de puntuación.
- Configuraciones específicas según el round:
  - **single-label**: Se instruyó a los modelos a asignar una puntuación de 0 a 4 en una sola dimensión (la más relevante) para cada commit, evitando asignar puntuaciones altas en múltiples dimensiones para el mismo commit. 
  - **few-shot**: Se proporcionaron ejemplos de commits anotados por humanos para guiar al modelo, seleccionados cuidadosamente para cubrir casos representativos de un acuerdo total, un acuerdo parcial y un desacuerdo total entre los anotadores humanos, con el objetivo de mejorar la capacidad del modelo para manejar casos ambiguos. Pueden encontrarse en `documentation/few-shot-examples.md`.
- El mensaje del commit sin incluir la línea "Fixes: " para evitar sesgar la anotación hacia BFC al igual que con los humanos
  - Si se selección una opción el **diff** se incluyó junto al mensaje, mostrando las líneas añadidas y eliminadas.
- Por último, se forzó al modelo a que respondiera con un JSON estructurado que incluyera las puntuaciones para cada dimensión (`bfc`, `bpc`, `prc`, `nfc`) junto a un breve razonamiento, una evaluación general de su comprensión del commit (`understanding`) y un resumen breve del commit (`summary`).

### 2.3.3 Procesamiento de los datos en bruto

Tras realizar la experimentación, se obtuvo un conjunto de anotaciones en bruto por modelo y configuración, almacenadas en `output/rX/` como archivos JSON (uno por cada commit/modelo/round). Estos archivos contienen la respuesta completa del modelo, incluyendo el razonamiento y la evaluación de comprensión.

Para poder comparar las anotaciones de los modelos con las de los humanos, se procesaron estos archivos JSON para extraer únicamente las puntuaciones asignadas a cada dimensión (BFC, BPC, PRC, NFC) y se consolidaron en archivos CSV por modelo y round. Estos CSVs contienen una fila por commit con las puntuaciones asignadas por el modelo, facilitando así su análisis posterior en la fase de análisis. El script `batch_convert_models_to_csv.py` se encargó de esta tarea de procesamiento y consolidación, generando los archivos CSV finales que se encuentran en `data/llm-annotator-results/rX/` para cada modelo y round.


### 3. Fase de análisis

En esta fase se compararán las anotaciones de los modelos con las anotaciones humanas utilizando métricas de acuerdo como Cohen's Kappa, Krippendorff's Alpha y alt-test.

La fase de análisis se documenta por completo en el Notebook `analysis/discussion.ipynb`, donde se presentan los resultados de las métricas de acuerdo para cada modelo y configuración.

PONER CONLUSIONEAS -> GPT-OSS