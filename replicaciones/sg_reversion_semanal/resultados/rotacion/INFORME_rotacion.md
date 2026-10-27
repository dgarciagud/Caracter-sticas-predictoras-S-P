# Reversión semanal simple: estadísticos y reducción de la rotación (exploratorio)

`python -m replicaciones.sg_reversion_semanal.rotacion`. Componentes del S&P 500 entre ene-2018 y dic-2021 (205 semanas).
- **Señal:** −retorno a 5 sesiones, ordenado dentro del sector.
- **Cartera:** quintiles largo/corto neutrales a sector, señal el miércoles y ejecución al cierre siguiente.
- **Rejilla:** 10 variantes fijadas antes de verlas e informadas todas. Con 10 variantes y t < 2, la mejor está
  sesgada al alza.

Tablas: `variantes.csv` y `estadisticos.json`. Figura: `rotacion.png`.

## 1. Estadísticos de la señal
- **IC medio:** 0,018 ejecutando al cierre de la señal (t 1,95; IR anual 0,99; positivo el 52 % de las semanas) y
  0,014 al día siguiente (t 1,6).
- **Decaimiento:** el IC cae a 0,011 a las 2 sesiones, a 0,009 a las 3 y a cero a las 4.
- **Deciles (exceso sobre el sector, anual):**
  - monótonos, del +8,2 % (mayores caídas) al −7,2 % (mayores subidas);
  - las dos patas aportan (larga +23,7 %, corta +10,8 % en bruto);
  - diferencial L/S de +12,9 % bruto.
- **Rendimiento L/S bruto (lag 1):** +12,9 % anual, volatilidad 17,7 %, Sharpe 0,73 (t 1,44), máxima caída
  −21 %, semanas positivas 53 %.
- **Asimetría 4,3 y curtosis 36:** la mejor semana da +21,8 % y la peor −8,2 %.
- **Por años:** 2018 +10,2 %, 2019 −7,1 %, 2020 +41,7 %, 2021 +5,8 %.
- **Sin las semanas de feb-abr de 2020:** +1,6 % anual. Casi todo el resultado es el rebote del crash del COVID.
- **Rotación:**
  - la correlación de rangos de una semana a la siguiente es 0,00;
  - sólo el 23 % del quintil superior repite;
  - rotación semanal 3,10 (|Δw| de ambas patas, sobre un máximo de 4), es decir, ≈ 161 pb de coste anual por
    cada pb por lado;
  - **coste de equilibrio: 8 pb por lado.**

## 2. Variantes para reducir la rotación

| Variante | Bruto | Rotación | Equilibrio (pb/lado) | Neto 5 pb | Neto 10 pb | Bruto sin feb-abr 2020 | Neto 5 pb sin feb-abr 2020 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Base (quintiles, semanal) | 12,9 % | 3,10 | 8,0 | 4,8 % | −3,3 % | 1,6 % | −6,4 % |
| Cuantil 10 % | 16,4 % | 3,44 | 9,1 | 7,4 % | −1,5 % | 2,6 % | −6,3 % |
| Cuantil 30 % | 11,6 % | 2,75 | 8,1 | 4,4 % | −2,7 % | 2,5 % | −4,6 % |
| Rebalanceo quincenal | 5,5 % | 3,12 | 3,4 | −2,6 % | −10,8 % | 7,0 % | −1,1 % |
| Banda: entra al 20 %, sale al 35 % | 11,8 % | 2,56 | 8,9 | 5,1 % | −1,5 % | 1,9 % | −4,7 % |
| Banda: entra al 20 %, sale al 50 % | 11,2 % | 2,03 | 10,6 | 5,9 % | +0,6 % | 2,9 % | −2,3 % |
| Parcial λ = 0,5 | 9,6 % | 1,68 | 11,0 | 5,2 % | +0,8 % | 5,4 % | +1,0 % |
| **Parcial λ = 0,25** | 8,0 % | **0,70** | **22,0** | **6,2 %** | **+4,4 %** | **6,5 %** | **+4,7 %** |
| Formación de 2 semanas | 4,3 % | 2,23 | 3,7 | −1,5 % | −7,3 % | 4,9 % | −0,9 % |
| Banda 50 % + parcial λ = 0,5 | 9,6 % | 1,34 | 13,8 | 6,1 % | +2,6 % | 6,8 % | +3,3 % |

**Rebalanceo parcial** (w = λ·objetivo + (1−λ)·w anterior):
- Es lo que más reduce la rotación y lo único que gana dinero neto con 10 pb y también sin el crash de 2020.
- Con λ = 0,25 la rotación cae un 77 % y el coste de equilibrio sube a 22 pb.
- Equivale a promediar las carteras de reversión de ~4 semanas. El rendimiento pasa así de la reversión a 1
  semana, rápida y concentrada en 2020, a una reversión a 1-4 semanas, más estable.
- Ninguna variante es estadísticamente significativa por separado (t bruto entre 0,5 y 1,6).

**Bandas de histéresis:** reducen la rotación un 17-34 % sin perder casi rentabilidad bruta. Es la forma más
barata de recortar costes manteniendo la señal.

**Cuantiles más extremos (10 %):** más rentabilidad bruta pero más rotación; el coste de equilibrio apenas mejora.

**Rebalancear cada dos semanas o alargar la formación a 2 semanas:** sin el crash rinden más que la base
(+5-7 %), pero con el crash pierden la semana del rebote y el equilibrio cae a 3-4 pb.

## 3. Otras palancas (no probadas aquí)
- **Ejecución:** subasta de cierre (MOC) o algoritmos VWAP. En grandes valores de EE. UU. el coste por lado puede
  quedar en 2-5 pb.
- **Optimización con coste explícito:** maximizar α − λ_coste·|Δw| en lugar de reglas por cuantiles. Es la
  generalización de bandas y rebalanceo parcial.
- **Universo líquido:** el quintil de mayor ADV tiene menos coste por operación.
- **Operar sólo en semanas de alta dispersión o VIX alto:** concentra la rotación cuando la señal paga más
  (ver `07_dispersion` de la réplica).
- **Combinar con señales lentas** (calidad, momentum de beneficios) o usar la reversión sólo como filtro de
  ejecución, como en la pág. 11 del paper y la sección 2.1 de la v3.
