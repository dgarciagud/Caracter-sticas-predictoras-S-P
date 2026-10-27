# Réplica SG · v3: fundamentales SEC, dividendos y sustitutos de las revisiones de BPA

**Conclusión.** Añadir los fundamentales que el paper sitúa entre los diez primeros por SHAP (rentabilidad por
dividendo, FCF/capital invertido, patrimonio sin intangibles/capitalización) y dos sustitutos de las
revisiones de analistas **mejora el modelo, pero no lo suficiente para replicar el paper**. El IC medio del
modelo se duplica (0,0065 → 0,0116, t = 2,7 sobre todas las sesiones) y la rotación baja un 17 %. El L/S
invertible pasa de −1,2 % a +1,1 % bruto y de −10,0 % a −6,2 % neto. Sigue lejos de la reversión simple
(+12,9 % bruto) y no gana neto de costes. Donde sí se acerca al paper es en dos cosas:
- **El ML como filtro de ejecución:** mejora las seis carteras factoriales mensuales (+0,4 a +2,2 puntos al
  año), con menos rotación y casi la misma exposición, como en la pág. 11.
- **La atribución:** el residuo no lineal pasa de −1,4 % a +0,8 % y aparecen calidad y dividendo como
  contribuyentes.

Mismo universo, modelo, cartera y análisis que la v2 ([`../../INFORME.md`](../../INFORME.md)). Sólo cambian las
características. Figuras en `figuras/` y tablas en esta carpeta.

## 1. Qué se añade y de dónde sale

| Variable SG (SHAP, pág. 3) | v3 | Fuente |
| --- | --- | --- |
| Dividend Yield | Dividendos en efectivo de los últimos 12 meses / precio | AlgoSeek (Jansen): dividendo = (1 − `adjustment_factor`) × cierre previo en cada `CashDiv`, fecha ex |
| Free Cash Flow to Capital Invested | (FCO − capex) TTM / (patrimonio + deuda a largo plazo) | SEC XBRL |
| Equity-Intangibles to Market Value | (patrimonio − fondo de comercio − intangibles) / capitalización | SEC XBRL + acciones en circulación (`dei`) |
| FY2 EPS Growth Forecast | Cambio interanual del BPA TTM / precio (crecimiento **pasado**) | SEC XBRL |
| 7/75/180-day EPS Revision | **SUE**: (BPA trimestral − BPA del mismo trimestre del año anterior) / desviación de ese cambio en 8 trimestres. **EAR**: retorno residual en [a−1, a+1], con a = la sesión de mayor volumen relativo de las 40 previas a la presentación del 10-Q/10-K. Días desde el anuncio | SEC XBRL + barras AlgoSeek |
| (otros factores de SG, págs. 14-15) | E/P TTM, ROE, beneficio bruto/activos, crecimiento de activos, log capitalización (tamaño real en lugar del ADV) | SEC XBRL |

**Datos point-in-time.**
- Se usa la API `companyfacts` de la SEC y, para cada periodo, el valor de su **primera** presentación.
- Cada dato se usa desde la sesión siguiente a la presentación.
- Los trimestres que sólo se publican acumulados (flujos de caja, 4.º trimestre) se obtienen por diferencia de
  acumulados con el mismo inicio de ejercicio.
- Un concepto que la empresa deja de usar caduca a los 400 días.

No se usa `xbrl_download.py` de Jansen. Su API Frames da cada dato según la *última* presentación y la fecha
por la última de ellas. Para la cuenta de resultados eso es a menudo el 10-Q del año siguiente: hasta 12 meses
de retraso.

**Mapa ticker → CIK** (`../../data_ref/cik_sp500.csv`, con la fuente de cada fila).
- **Candidatos:**
  - el histórico git de [datasets/s-and-p-500-companies](https://github.com/datasets/s-and-p-500-companies);
  - `company_tickers.json` de la SEC;
  - y, en último caso, la búsqueda por nombre en EDGAR.
- **Validación:** un candidato vale si la empresa presentó 10-Q/10-K al principio y al final del periodo de la
  acción en el índice, con activos ≥ 500 M$. Eso descarta tickers reutilizados, SPAC y escisiones (HCP → Quality
  Care Properties).
- **Cobertura:** 613 de 638 símbolos (96,7 % de las observaciones). Faltan sobre todo reorganizaciones
  societarias que cambian de CIK a mitad de periodo (DIS, CI, AVGO, DOW, DD, FOX, IR, XRX, WRK), que quedan como
  valores ausentes.
- **Cobertura de cada variable en el panel:** 87-97 %, salvo beneficio bruto/activos (42 %: muchas empresas no
  etiquetan `GrossProfit`).

**Lo que sigue sin poder replicarse.** Las revisiones de analistas de verdad, a 7, 75 y 180 días, y las
previsiones de consenso. La SUE y el EAR recogen la reacción a los resultados, que es cuando se concentran las
revisiones, pero no los cambios de opinión entre publicaciones.

## 2. Resultados frente a la v2

| | v2 (precio y volumen) | **v3** |
| --- | --- | --- |
| IC medio del modelo (todas las sesiones; t) | 0,0065 (1,7) | **0,0116 (2,7)** |
| IC medio, miércoles | 0,0111 | 0,0155 (reversión 1 semana: 0,0235) |
| ML bruto, lag 0 | +0,6 % | +0,7 % |
| ML bruto, lag 1 | −1,2 % | **+1,1 %** |
| ML neto 10 pb, lag 1 | −10,0 % | **−6,2 %** (Sharpe −0,53) |
| Rotación semanal | 1,71 | 1,41 |
| Residuo de la atribución (% anual) | −1,4 | **+0,8** |
| Beta frente a SPY | 0,01 | −0,00 |

Por años (neto, lag 1): 2018 −2,3 %; 2019 −12,9 %; 2020 −5,0 %; 2021 −4,4 % (`01_rendimiento_anual.csv`).

**SHAP (`03_shap.png`).**
- **Último reentreno (oct-2021):** crecimiento de activos (+), retorno a 1 semana frente al sector (signo de
  reversión, −0,96), E/P (−), rango a 1 mes (−), crecimiento del BPA (−), momentum 3m1m (+), patrimonio tangible
  / capitalización, SUE (+0,83), beta (+) y ROE (+).
- **Media de todos los reentrenos:** crecimiento de activos, retorno 1 semana, momentum 3m1m, E/P, FCF/IC, SUE,
  volatilidad, crecimiento del BPA y patrimonio tangible.
- **Frente al paper:**
  - Como en SG, la reversión es central y entran FCF/IC, patrimonio tangible y el sustituto de las revisiones,
    con su signo (SUE alto, más retorno esperado).
  - A diferencia de SG, la rentabilidad por dividendo apenas pesa.
  - Los signos de E/P y de crecimiento de activos (prefiere empresas caras y que crecen) reflejan el periodo
    2018-2021, dominado por el crecimiento, más que una relación estable.

**Exposiciones medias (z, larga − corta).**
- Reversión +0,63; momentum 12m +0,68; momentum 3m +0,62.
- Calidad +0,68; baja volatilidad +0,48.
- SUE +0,38; EAR +0,25.
- Tamaño pequeño −0,49; dividendo −0,38; valor −0,13.

El perfil se parece al del paper: reversión con sesgo a momentum de beneficios y de precio, y calidad positiva.

**Atribución (`04_atribucion_anual.csv`, bruto lag 1, % anual).**
- Positivas: reversión +1,8, calidad +1,2, dividendo +1,0 (la exposición negativa a un factor que perdió),
  residuo +0,8, tamaño +0,4.
- Negativas: momentum 12m −1,9, momentum 3m −0,8, baja beta −0,7.
- SUE y EAR: −0,2 cada una.

El paper obtiene un residuo de +3,3 y una reversión de +2,4, con las revisiones a 1 semana y 3 meses en
positivo. Los sustitutos de las revisiones no aportan lo que SG atribuye a las revisiones reales.

**Atribución académica (`11_atribucion_academica.csv`, nuevo).** Regresión semanal del ML neto sobre
Fama-French 5 + momentum (errores HAC):

| Factor | Carga | t |
| --- | --- | --- |
| SMB (tamaño) | −0,27 | −4,5 |
| MOM (momentum) | +0,26 | 5,2 |
| HML (valor) | −0,14 | −2,3 |
| CMA (inversión) | −0,24 | −1,9 |

- R² = 0,64.
- Alfa neto: −8,9 % anual (t −2,6). Los costes medios son ≈ 7,3 %, así que el alfa bruto ronda −1,6 %.
- En la regresión mensual con QMJ (AQR, EE. UU.), la carga a calidad es +0,41 (t 2,0).
- Dos tercios de la varianza del L/S son factores conocidos: momentum, empresas grandes y crecimiento.
- La reversión simple, en cambio, apenas carga en factores (R² 0,14): su alfa bruto ≈ +9 % anual se lo comen
  los costes.

**Otros cortes.**
- **VIX:** sin patrón claro. Quintil más alto +7,0 %, bajo +8,5 %, alto −29 %.
- **Deciles del SPY:** sin la U del paper.
- **Dispersión:** como en la v2, el ML neto pierde mucho con dispersión baja (−17,8 %) y casi nada con dispersión
  alta (+0,2 %).
- **Liquidez:** sólo el quintil más líquido es positivo (+3,6 %; +14,7 % en 2021).
- **Lags:** con señal el miércoles, el ML bruto gana +0,7 % con lag 0, +1,1 % con 1, +1,2 % con 2, +2,0 % con 3 y
  +1,3 % con 4. Con fundamentales la señal es más lenta y ya no decae en un día, pero tampoco es la señal rápida
  de reversión del paper.

### 2.1 El ML como filtro de ejecución (`09_filtro_ml_factores.csv`, `09_filtro_ml.png`)

Mismos factores que la pág. 11 del paper. El multifactor combina valor, momentum y calidad. Carteras mensuales
neutrales a sector, netas de 10 pb.

| Factor | Sin filtro | Con filtro ML | Mejora | Rotación (sin → con) |
| --- | --- | --- | --- | --- |
| Multi-Factor | −2,6 % | −1,5 % | +1,2 | 0,75 → 0,67 |
| Value | −1,5 % | −1,2 % | +0,4 | 0,59 → 0,54 |
| Momentum | −6,1 % | −4,7 % | +1,4 | 1,10 → 0,91 |
| Quality | +1,3 % | +2,0 % | +0,8 | 0,50 → 0,44 |
| Low Vol | −5,9 % | −5,1 % | +0,8 | 0,84 → 0,73 |
| Growth | −5,7 % | −3,6 % | +2,2 | 0,73 → 0,67 |

- El filtro mejora **todos** los factores. Con la v2 sólo mejoraba tres de cinco, y por menos.
- Baja la rotación entre un 8 y un 17 % y la exposición al factor apenas cambia (`exposicion_z`).
- El multifactor filtrado gana un 4,7 % acumulado en cuatro años frente al no filtrado. El paper obtiene un 10 %
  en veinte.
- Es el resultado del paper que mejor se replica. La razón probable: para filtrar basta con que la señal ordene
  algo mejor que el azar y no tiene que pagar su propia rotación.

## 3. Lectura

1. **Los fundamentales y los sustitutos de las revisiones duplican la capacidad predictiva del modelo,** pero
   el IC sigue por debajo del de la reversión simple, y los árboles mezclan la reversión con apuestas de estilo
   (momentum, crecimiento, empresas grandes) que explican dos tercios de su varianza. El paper no publica cuánto
   pesan esas apuestas en su modelo.
2. **Los sustitutos no reproducen el papel de las revisiones.** La SUE entra en el modelo con el signo esperado,
   pero su IC aislado es pequeño (0,009) y su contribución lineal, negativa. El EAR no predice: −0,016 incluso
   con anuncios recientes. La tesis de SG (la reversión condicionada a revisiones de analistas) sigue sin poder
   contrastarse con datos públicos.
3. **Como filtro de ejecución, la señal sí es útil,** con menos rotación y la misma exposición, en línea con el
   paper.
4. **Muestra corta:** con 205 semanas, las diferencias v2–v3 de rendimiento (≈ +2-4 puntos anuales) están dentro
   del error típico (±5-11 puntos). La mejora del IC (t de 1,7 a 2,7) y la del filtro, en los seis factores, son
   más sólidas que las de rendimiento.
5. Esta v3 se diseñó antes de ver sus resultados, con la misma configuración del modelo que la v2, y no se ha
   ajustado después.

## 4. Reproducir

```bash
bash scripts/00_descargar_datos.sh                           # precios y ETFs (Jansen)
python ../ml4t/machine-learning-for-trading/data/factors/ff_download.py    # Fama-French (con PYTHONPATH/ML4T_DATA_PATH del repo de Jansen)
python ../ml4t/machine-learning-for-trading/data/factors/aqr_download.py   # QMJ de AQR
export SEC_USER_AGENT="tu-proyecto tu@email"                 # la SEC exige un User-Agent con contacto
python -m replicaciones.sg_reversion_semanal.run             # v3 (por defecto); ~5 min la primera vez (descarga SEC, ~600 empresas)
SG_VERSION=v2 python -m replicaciones.sg_reversion_semanal.run   # v2
```
