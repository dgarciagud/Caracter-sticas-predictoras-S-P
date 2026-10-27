# Réplica · SG Global Style Counselling, «Monetising noise through cross-sectional stock mean-reversion» (13-05-2026)

**Conclusión.** Con los datos públicos del repositorio (componentes del S&P 500, 2017-2021, sólo precio y
volumen) **el resultado principal del paper no se replica**: el modelo ML semanal no bate a la reversión de
1 semana ni antes de costes (+0,6 % anual bruto con ejecución al cierre de la señal, frente a +14,4 % de la
reversión) y ambos pierden dinero netos de 10 pb por lado (−10,0 % y −3,3 % anual con ejecución al día
siguiente). Sí se reproducen los hechos estilizados de los que parte el paper: reversión fuerte a 1 semana,
rentabilidad concentrada en semanas de mercado extremo y alta volatilidad (2020), decaimiento rápido de la
señal con el lag de ejecución y el ciclo reversión–momentum por periodo de formación. La diferencia más
probable es la ausencia de los fundamentales (revisiones de BPA, crecimiento, dividendo), que según el propio
paper condicionan qué caídas revierten, y la muestra mucho más corta (1-4 años de entrenamiento frente a 10).

**v3** ([`resultados/v3/INFORME_v3.md`](resultados/v3/INFORME_v3.md)). Añade dividendos, fundamentales XBRL de
la SEC (point-in-time) y dos sustitutos de las revisiones de BPA (SUE y retorno en torno a los resultados).
- El IC del modelo se duplica, de 0,0065 a 0,0116.
- El L/S invertible pasa a +1,1 % bruto y −6,2 % neto.
- El filtro de ejecución mejora las seis carteras factoriales, como en el paper.

Sigue sin batir a la reversión simple. Las revisiones de analistas reales siguen sin poder replicarse.

Este informe describe la **v2** (sólo precio y volumen). Se reproduce con `SG_VERSION=v2`.

## 1. Qué hace el paper

- Modelo ML (marco de SG de 2017) que predice el retorno de cada acción a 1 semana con ~decenas de factores,
  entrenado globalmente con ventana móvil de 10 años; L/S neutral a sector (y región), neto de costes,
  2006-2026. Índices en vivo: US (SGEPMRU), Europa (SGEPMREU), Global 60/40 (SGEPMRWU).
- Hechos que reporta: rendimiento en forma de U por decil del mercado (16,6 % y 16,7 % anual en los deciles
  extremos) y máximo con el VIX más alto (18,8 %); SHAP dominado por el retorno a 1 semana (reversión) y
  revisiones de BPA a 7/75/180 días, crecimiento FY2 y rentabilidad por dividendo; exposiciones positivas a
  reversión, momentum de beneficios y momentum de precio; atribución: residuo ≈ 3,3 %, reversión ≈ 2,4 %;
  Europa peor desde 2023 por menor dispersión, decaimiento más rápido y liquidez; el ML como filtro de
  ejecución mejora todas las carteras factoriales mensuales sin perder exposición.

## 2. Datos y desviaciones (lo que no se puede replicar)

| Paper | Aquí | Consecuencia |
| --- | --- | --- |
| US + Europa (FTSE), 2006-2026 | Componentes del S&P 500 día a día (AlgoSeek, `load_sp500_daily_bars`), 2017-2021; fuera de muestra 2018-01 a 2021-12 (205 semanas) | No hay Europa ni el análisis EE. UU.–Europa; la muestra es corta y la domina 2020 |
| Factores SG con FactSet: revisiones de BPA, crecimiento FY1/FY2, dividendo, FCF/IC, capital-intangibles/valor | Sólo precio y volumen: 17 características de la lista de factores de SG (págs. 14-15) que se pueden calcular con precios (retornos a 1 semana/1 mes/3m1m/12m1m, residuales, beta, volatilidades, rango, ADV, shock de volumen, distancia al máximo, retorno del sector) | Falta justo lo que el paper presenta como el condicionante de la reversión |
| Tamaño (capitalización) | log ADV a 6 meses como proxy | — |
| Sectores GICS | ETF SPDR cuyo exceso sobre SPY más correla con el residuo de la acción (252 sesiones, point-in-time) | Asignación aproximada (p. ej., AMZN→XLK) |
| Ventana móvil de 10 años | Ventana expansiva, reentreno trimestral, mín. 189 sesiones, embargo 6 | El primer modelo se entrena con 9 meses |
| Costes de SG (no publicados) e índice con controles de rotación | 10 pb por lado × rotación (|Δw| de ambas patas), sin amortiguar la rotación; sensibilidad 0/5/10/20 pb | El neto es sólo orientativo |
| MSCI World | SPY | — |

Las salidas del índice: si una acción deja de cotizar en las barras, se usa su último precio (hasta 5 sesiones).

## 3. Método

- `datos.py` reutiliza `linea1a.datos` (`SECTORES`, `_fred` para VIXCLS) y los loaders del repositorio de
  Jansen (`load_sp500_daily_bars`, `load_etfs`); precio total-return = close × adj_factor.
- Objetivo: rango transversal del retorno de t a t+5 sesiones en exceso de la media del sector.
  Características en rango transversal. LightGBM walk-forward (ver `config.py`).
- Cartera: quintil superior largo y quintil inferior corto dentro de cada sector, pesos iguales, 1 de nominal por
  pata; señal el miércoles al cierre (como en el análisis de lags del paper); ejecución al cierre siguiente
  (lag 1) en la versión invertible y al cierre de la señal (lag 0) en la teórica.
- «Reversión 1 semana»: la misma cartera ordenando por −retorno a 1 semana dentro del sector (la estrategia
  básica del paper).
- Retorno anualizado = media semanal × 52 (aritmético, como las barras del paper).
- Nada se escribe en `resultados/trials.jsonl`: la réplica no importa `linea1a.registro`.

### Iteraciones (todas fuera de muestra; se informan para no ocultar grados de libertad)

| Versión | Cambio | ML bruto lag 0 | ML neto lag 1 | IC medio |
| --- | --- | --- | --- | --- |
| v0 | Incluía VIX y retorno del mercado como características; objetivo sin ajustar por sector | −2,3 % | −11,5 % | — |
| v1 | Sin variables de mercado (con un objetivo transversal sólo sirven para que los árboles identifiquen fechas: eran las 2 primeras por SHAP) y objetivo en exceso del sector, coherente con la cartera | +2,2 % | −11,3 % | 0,0066 |
| **v2 (final)** | Árboles poco profundos (7 hojas, profundidad 3, hojas ≥ 2.000 obs.), como `LGBM_PARAMS` de linea1a, por sobreajuste de v1 | **+0,6 %** | **−10,0 %** | 0,0065 |

La reversión de 1 semana tiene un IC medio de 0,016 en el mismo periodo. Se fijó v2 antes de verla y se
mantiene aunque no mejore; no se probaron más variantes.

## 4. Resultados (figuras en `resultados/figuras/`, tablas en `resultados/`)

### 4.1 Rendimiento (`01_rendimiento.csv`, `01_rendimiento.png`)

| Estrategia | Lag | Coste (pb) | Ret. anual | Vol. | Sharpe | Máx. DD | Rotación semanal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ML semanal | 0 | 0 | 0,6 % | 11,4 % | 0,05 | −18 % | 1,71 |
| ML semanal | 1 | 0 | −1,2 % | 10,7 % | −0,11 | −19 % | 1,71 |
| ML semanal | 1 | 10 | −10,0 % | 10,7 % | −0,94 | −38 % | 1,71 |
| Reversión 1 semana | 0 | 0 | 14,4 % | 21,4 % | 0,67 | −29 % | 3,10 |
| Reversión 1 semana | 1 | 0 | 12,9 % | 17,7 % | 0,73 | −21 % | 3,10 |
| Reversión 1 semana | 1 | 10 | −3,3 % | 17,7 % | −0,18 | −38 % | 3,10 |

Por años (`01_rendimiento_anual.csv`), la reversión bruta lag 0 gana en 2018 (+11 %), 2020 (+48 %, casi todo
marzo-abril) y 2021 (+9 %), y pierde en 2019 (−11 %). El paper dice que el ML gana netos de costes; aquí la
rotación (171-310 % semanal sobre 2 de nominal) hace que 10 pb por lado cuesten 9-16 puntos al año: sin
control de rotación, ninguna de las dos estrategias es invertible con estos datos.

### 4.2 Regímenes (`02_*.csv`, `02_regimenes.png`)

ML neto: único quintil de VIX positivo, el más alto (+2,4 %, frente a −2 % a −27 % en el resto), en línea
con el paper (máximo en VIX alto), pero sin la U por decil del mercado (deciles 1 y 10: −9,8 % y −24 %). La
reversión simple sí concentra su rentabilidad en las semanas extremas de 2020.

### 4.3 Qué mueve la señal (`03_shap.png`, `03_shap_medio_abs.csv`)

En el último reentreno (oct-2021) la primera variable por SHAP es el retorno a 1 semana frente al sector, con
signo de reversión (corr. valor–SHAP −0,95), como en el paper. Le siguen beta (+), rango 1 mes (−), momentum
3m1m (+), ADV (+) y volatilidad (−). Promediando todos los reentrenos el orden es momentum 3m1m, ADV,
volatilidad 6m, momentum residual 12m1m y retorno 1 semana: el modelo reparte peso entre efectos de
momentum y volatilidad que en 2018-2021 no fueron estables.

### 4.4 Exposiciones y atribución (`04_*.csv`, `04_exposiciones_atribucion.png`, `04_atribucion_serie.png`)

Exposiciones medias (z, larga − corta): reversión 1 semana +0,70, momentum 12m +0,76, momentum 3m +0,73,
baja volatilidad +0,49, tamaño pequeño −0,45. Como en el paper, el modelo es reversión a corto plazo con
sesgo a momentum de medio plazo. Atribución (bruto lag 1, % anual): reversión +2,1 (el paper, +2,4),
ADV pequeño +0,5, baja vol. +0,3, momentum 12m −1,8, momentum 3m −0,7 y **residuo −1,4** (el paper, +3,3).
Es decir, la reversión aporta lo mismo que en el paper; lo que falta es el componente no lineal, y el sesgo a
momentum restó en un periodo con giros bruscos de momentum (nov-2020).

### 4.5 Patas, día de la semana y lags (`05_estacionalidad_lags.csv`, `05_estacionalidad_lags.png`)

- Reversión, bruto lag 0: pata larga vs mercado +7,4 %, mercado vs pata corta +7,0 % (las dos aportan, como
  en el gráfico del paper para Europa).
- Por día de señal: reversión de +5,0 % (lunes) a +15,2 % (jueves): estacionalidad marcada, frente al «limited
  seasonality» del paper. ML: −3,9 % a +0,6 %, sin día ganador.
- Lags (señal el miércoles, ML bruto): +0,6 % (lag 0), −1,2 % (1), −2,2 % (2), −1,1 % (3), −1,7 % (4). La
  señal se agota en un día, como el decaimiento acelerado que el paper describe para Europa.

### 4.6 Liquidez (`06_por_quintil_adv.csv`, `06_liquidez.png`)

ML neto por quintil de ADV (2018-2021 / 2021): Top −2,3 % / +10,8 %; 2: −17,7 % / −11,2 %; 3: −23,8 % /
−18,5 %; 4: −12,5 % / −18,8 %; Bottom −12,1 % / −12,3 %. No aparece el patrón del paper (en EE. UU., positivo
en todos los quintiles); dentro del S&P 500 todas las acciones son líquidas y el quintil «Bottom» no es
comparable con el de SG.

### 4.7 Dispersión (`07_dispersion.csv`, `07_dispersion.png`)

La mediana del |retorno semanal| y la dispersión transversal del S&P 500 repiten el perfil del gráfico de EE. UU.
del paper (pico en 2020 y repunte en 2021). El ML neto gana más con dispersión previa alta (−2,3 %) que baja
(−21,5 %), en la dirección del argumento del paper.

### 4.8 Ciclo reversión–momentum (`08_ciclo_momentum.csv`, `08_ciclo_momentum.png`)

Decil superior − inferior por retorno pasado, 1 semana de tenencia, bruto, 2018-2021: formación 1 semana
−22,7 % (el paper: ≈ −21 %, 1990-2024), y desde ahí valores mixtos (1-2 s: +3,1 %; 3-4 s: −17,2 %; 12-18 s:
+2,3 %; 18-24 s: +3,9 %; 24-48 s: −3,6 %). La reversión a 1 semana se replica casi exacta; la joroba de
momentum a medio plazo no se ve en 4 años.

### 4.9 El ML como filtro de ejecución (`09_filtro_ml_factores.csv`, `09_filtro_ml.png`)

Carteras mensuales neutrales a sector (quintiles, 10 pb). Regla: no se añade una larga nueva con puntuación ML en
el quintil inferior ni se vende una larga con puntuación en el superior (simétrico en la corta).

| Factor | Sin filtro | Con filtro ML | Exposición z (sin → con) | Rotación (sin → con) |
| --- | --- | --- | --- | --- |
| Multi-Factor (momentum 12m + baja vol.) | −4,2 % | −4,4 % | 1,61 → 1,57 | 0,99 → 0,89 |
| Momentum 12m1m | −6,1 % | −5,6 % | 2,21 → 2,13 | 1,10 → 0,95 |
| Momentum 3m1m | −2,3 % | −2,8 % | 2,31 → 1,85 | 2,27 → 1,79 |
| Baja volatilidad | −5,9 % | −5,5 % | 1,92 → 1,88 | 0,84 → 0,74 |
| Baja beta | −5,5 % | −5,4 % | 1,76 → 1,72 | 0,73 → 0,67 |

El filtro baja la rotación un 10-20 % y conserva casi toda la exposición, como en el paper, pero la mejora de
retorno es pequeña (+0,0 a +0,6 puntos en tres factores, empeora en dos): con una señal ML débil el filtro no
añade la mejora consistente que reporta SG. (Value, Quality y Growth no se pueden construir sin fundamentales.)

### 4.10 Diversificación (`10_beta_movil.csv`, `10_beta_movil.png`)

Beta del ML L/S frente a SPY: 0,01 (corr. 0,02); la beta móvil a 6 meses oscila en torno a cero, como el
índice de SG.

## 5. Lectura

1. La materia prima del paper existe en estos datos: la reversión semanal en el S&P 500 da +13-14 % bruto al año,
   las dos patas aportan y la señal decae en 1 día.
2. Lo que no se replica es el valor añadido del ML. Sólo con precios, el modelo aprende la reversión (primera por
   SHAP) pero la diluye con momentum, volatilidad y beta, y su IC es menos de la mitad del de la reversión simple.
   La tesis del paper es que la reversión condicionada a fundamentales (revisiones, crecimiento, rentabilidad)
   es la que funciona; sin esos datos esa tesis no se puede contrastar, sólo constatar que sin ellos no aparece.
3. Netos de costes, nada sobrevive sin controlar la rotación. SG publica índices netos positivos; su modelo de
   costes y su construcción (amortiguación de la rotación, restricciones de liquidez) no son públicos.
4. El periodo 2018-2021 es corto y está dominado por marzo-abril de 2020; cualquier cifra anual tiene un error
   típico de ±5-11 puntos (volatilidad anual / √4).

Para una réplica completa harían falta precios 2006-2026 de EE. UU. y Europa con pertenencia histórica a los
índices y estimaciones de consenso point-in-time (FactSet/IBES).

## 6. Reproducir

```bash
bash scripts/00_descargar_datos.sh        # el paso de COT puede fallar (cot_reports no compila en Python 3.13); esta réplica no lo usa
SG_VERSION=v2 python -m replicaciones.sg_reversion_semanal.run   # v2: ~2 min con 4 núcleos; caché en replicaciones/sg_reversion_semanal/cache/
python -m replicaciones.sg_reversion_semanal.run                # v3 (ver resultados/v3/INFORME_v3.md)
```
