# MiniGPT Educativo

Proyecto académico que implementa el pipeline:

**documentos → corpus → tokenización → vocabulario → IDs → secuencias/batches → embeddings → Transformer → logits → loss → backpropagation → modelo → generación**

## Estructura

```text
MiniGPT/
├── corpus/                  # AQUÍ colocan TXT/MD/JSON/PDF
├── datos/                   # corpus limpio, vocabulario y tokens generados
├── modelos/                 # checkpoint final minigpt.pt
├── resultados/              # historial de pérdidas
├── config.py                # parámetros del experimento
├── preparar_corpus.py       # documentos -> corpus limpio
├── tokenizador.py           # corpus -> tokens -> vocabulario -> IDs
├── dataset.py               # IDs -> secuencias X/Y y batches
├── modelo.py                # embeddings + Transformer causal
├── entrenar.py              # loss, backpropagation, AdamW y guardado
├── generar.py               # generación autoregresiva
├── chat.py                  # interfaz tipo chat en consola
├── inspeccionar.py          # herramienta docente para visualizar el pipeline
├── pipeline.py              # ejecuta preparación + tokenizer + entrenamiento
├── utilidades.py
├── requirements.txt
└── README.md
```

## 1. Corpus

Coloca documentos dentro de `corpus/`.

Soporta:
- `.txt`
- `.md`
- `.json`
- `.pdf` con texto seleccionable

Para la primera práctica se recomienda TXT o Markdown. PDF escaneado como imagen no funciona porque no se incluye OCR.

El JSON se recorre de manera recursiva y se extraen todos los valores de texto.

## 2. Instalación

### macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### Windows PowerShell
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

## 3. Parámetros recomendados

En `config.py` ya vienen estos valores:

```python
LONGITUD_CONTEXTO = 64
TAMANIO_BATCH = 16
DIMENSION_EMBEDDING = 128
NUM_CABEZAS = 4
NUM_CAPAS = 4
DIMENSION_FEED_FORWARD = 512
DROPOUT = 0.10
EPOCAS = 20
TASA_APRENDIZAJE = 3e-4
```

`DIMENSION_EMBEDDING` debe ser divisible por `NUM_CABEZAS`.

### Equipo modesto
```python
LONGITUD_CONTEXTO = 32
TAMANIO_BATCH = 8
DIMENSION_EMBEDDING = 64
NUM_CABEZAS = 4
NUM_CAPAS = 2
DIMENSION_FEED_FORWARD = 256
EPOCAS = 10
```

### Equipo con más capacidad
```python
LONGITUD_CONTEXTO = 128
TAMANIO_BATCH = 32
DIMENSION_EMBEDDING = 256
NUM_CABEZAS = 8
NUM_CAPAS = 6
DIMENSION_FEED_FORWARD = 1024
EPOCAS = 30
```

El código detecta automáticamente CUDA, Apple Silicon/MPS o CPU.

## 4. Ejecutar todo

Después de colocar los documentos en `corpus/`:

```bash
python pipeline.py
```

o en macOS:

```bash
python3 pipeline.py
```

El pipeline ejecuta:

```text
corpus/
  ↓
preparar_corpus.py
  ↓
datos/corpus_limpio.txt
  ↓
tokenizador.py
  ↓
datos/vocabulario.json + datos/tokens.pt
  ↓
dataset.py
  ↓
X/Y + batches
  ↓
modelo.py
  ↓
embeddings + self-attention causal + feed-forward + Transformer
  ↓
entrenar.py
  ↓
modelos/minigpt.pt
```

## 5. Qué hace cada .py

### `config.py`
Parámetros y rutas.

### `preparar_corpus.py`
Une TXT, MD, JSON y PDF en un solo corpus limpio.

### `tokenizador.py`
Tokenizador educativo por palabras/signos. Construye vocabulario y asigna Token IDs. Incluye `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`.

### `dataset.py`
Genera ejemplos de predicción del siguiente token:

```text
X = [el, gato, come]
Y = [gato, come, pescado]
```

También construye batches.

### `modelo.py`
Es el corazón. Implementa:
- token embeddings;
- positional embeddings;
- Multi-Head Self-Attention causal;
- feed-forward;
- residual connections;
- LayerNorm;
- varios bloques Transformer;
- salida de logits sobre el vocabulario.

La clase principal es `MiniGPT`.

### `entrenar.py`
Hace forward, cross entropy loss, backpropagation, AdamW, gradient clipping y guarda el checkpoint.

### `generar.py`
Generación autoregresiva con temperature y top-k.

### `chat.py`
Interfaz educativa tipo chat. No convierte el modelo en ChatGPT real; simplemente usa el formato `Usuario:` / `Asistente:`.

### `inspeccionar.py`
Para mostrar en clase: texto -> tokens -> IDs -> batch -> X/Y.

### `pipeline.py`
Automatiza preparación, tokenización y entrenamiento.

## 6. Generación

```bash
python generar.py
```

Ejemplo de prompt:

```text
El servidor Apache
```

## 7. Chat

```bash
python chat.py
```

Para que aprenda mejor el formato conversacional, agrega al corpus ejemplos como:

```text
Usuario: ¿Qué es Apache?
Asistente: Apache es un servidor web.

Usuario: ¿Qué es SSH?
Asistente: SSH es un protocolo de acceso remoto seguro.
```

## 8. Tamaño del corpus

Para demostrar el pipeline, unos cientos de KB pueden bastar. Para observar algo más interesante, 1–5 MB de texto es mejor. Un corpus pequeño puede memorizar y generar texto incoherente; eso es útil para estudiar overfitting.

## 9. Qué deben entregar los estudiantes

- corpus usado y sus fuentes;
- tamaño del corpus;
- tamaño del vocabulario;
- número de tokens;
- longitud de contexto;
- dimensión de embeddings;
- número de heads;
- número de capas;
- número de parámetros;
- epochs;
- loss inicial/final;
- ejemplos de generación;
- análisis de overfitting y limitaciones.

## 10. Advertencia

Este modelo es exclusivamente educativo. No debe usarse para diagnóstico médico, asesoría legal, financiera ni decisiones de alto impacto.

## 11. Flujo que deben poder explicar

```text
DOCUMENTOS
↓
CORPUS
↓
TOKENIZADOR
↓
VOCABULARIO
↓
TOKEN IDs
↓
SECUENCIAS
↓
BATCHES
↓
EMBEDDINGS
↓
SELF-ATTENTION
↓
TRANSFORMER
↓
LOGITS
↓
SOFTMAX / NEXT TOKEN
↓
LOSS
↓
BACKPROPAGATION
↓
PESOS ACTUALIZADOS
↓
MODELO ENTRENADO
↓
GENERACIÓN AUTOREGRESIVA
```
