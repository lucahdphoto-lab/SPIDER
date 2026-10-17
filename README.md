# 🕷️ SPIDER
### **System for Portfolio, Index, and Data Economic Relationships**
*Piattaforma Istituzionale di Monitoraggio Macroeconomico e Intermarket in Tempo Reale su Grafo Radiale Concentrico*

---

## 📌 Panoramica del Progetto

**SPIDER** è un'applicazione quantitativa avanzata in Python progettata per monitorare, collegare e analizzare in tempo reale i rapporti tra variabili macroeconomiche istituzionali e asset finanziari globali.

Il mercato viene modellato come una **tela di ragno interattiva (Radial Network Graph)**:
- Al centro risiede il **Centro di Controllo (`SPIDER_CORE`)**, che aggrega lo stato sistemico del mercato, la matrice dei regimi economici (Goldilocks, Reflazione, Stagflazione, Deflazione) e il bias di politica monetaria.
- Sulla periferia sono disposti **30 Nodi Istituzionali Certificati**, distribuiti su **3 Anelli Concentrici** (Politica Monetaria, Fondamentali Macro, Asset Intermarket) e **4 Quadranti Radiali** (*Inflazione*, *Crescita*, *Lavoro*, *Liquidità/Rischio*).
- I nodi sono uniti dai **Fili della Tela**: archi matematici che misurano in tempo reale la forza, la direzione e la tensione/rottura delle correlazioni storiche mediante metriche basate su Z-Score rolling.

---

## 🏛️ Vincolo di Certificazione e Fonti Dati Istituzionali

Tutti i dati provengono **esclusivamente da API e database istituzionali e ufficiali**:
1. **Federal Reserve Bank of St. Louis (FRED):**
   - Inflazione USA (CPI, Core PCE, 5Y5Y Forward Inflation Expectations).
   - Mercato del Lavoro (Unemployment Rate U3, Non-Farm Payrolls, Initial Jobless Claims).
   - Crescita (Real GDP QoQ Annualized, Industrial Production, Advance Retail Sales).
   - Politica Monetaria & Tassi (Fed Funds Effective Rate, US Treasury 2Y, 10Y, 10Y TIPS Real Yield, Curva 10Y-2Y, High Yield Spread OAS).
2. **Banca Centrale Europea (ECB Data Portal) & Eurostat:**
   - Tassi di riferimento BCE (Deposit Facility Rate DFR).
   - Inflazione Eurozona (HICP) e rendimenti governativi europei (Bund 10Y, BTP 10Y e Spread BTP-Bund).
3. **Mercati Finanziari Real-Time / CME / CBOE / Yahoo Finance:**
   - Indici Azionari Benchmark: S&P 500 (`^GSPC`), Nasdaq 100 (`^NDX`), Russell 2000 (`^RUT`), DAX 40 (`^GDAXI`), FTSE MIB (`FTSEMIB.MI`).
   - Volatilità & Rischio: CBOE Volatility Index (`^VIX`).
   - Valute: US Dollar Index (`DX-Y.NYB`), Cambio EUR/USD (`EURUSD=X`).
   - Materie Prime: Oro Future (`GC=F`), Petrolio WTI (`CL=F`), Rame Future "Dr. Copper" (`HG=F`).

Ogni singolo dato traccia l'**Ente Certificatore**, il **Ticker Ufficiale** e il **Timestamp esatto** dell'ultimo rilascio, garantendo piena conformità e trasparenza istituzionale.

---

## 🕸️ Architettura della Tela: I 30 Nodi Periferici

### Anello 1 (Interno): Politica Monetaria, Tassi & Rendimenti
| ID Nodo | Nome Indicatore | Ticker Ufficiale | Fonte Certificata | Frequenza | Ruolo nella Tela |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `RATE_FED` | Fed Funds Effective Rate | `DFF` | FRED / Federal Reserve | Giornaliera | Tasso guida globale del costo del denaro USA |
| `RATE_ECB` | ECB Deposit Facility Rate | `ECBDFR` | ECB / FRED | Giornaliera | Tasso guida della politica monetaria Eurozona |
| `YIELD_US02` | US Treasury 2Y Yield | `DGS2` | FRED / US Treasury | Giornaliera | Termometro delle aspettative sui tassi FED a 12-24 mesi |
| `YIELD_US10` | US Treasury 10Y Yield | `DGS10` / `^TNX` | FRED / CME | Real-Time / Daily | Benchmark globale del tasso di sconto di tutti gli asset |
| `YIELD_REAL10` | US 10Y TIPS (Real Yield) | `DFII10` | FRED / US Treasury | Giornaliera | Rendimento reale al netto dell'inflazione; barometro Tech/Oro |
| `CURVE_10_2` | Spread Curva USA 10Y-2Y | `T10Y2Y` | FRED | Giornaliera | Indicatore primario di inversione della curva e ciclo recessivo |
| `YIELD_DE10` | Bund Tedesco 10Y | `IRLTLT01DEM156N` | FRED / Bundesbank / ECB | Giornaliera | Benchmark privo di rischio per l'area Euro |
| `SPREAD_BTP` | Spread BTP 10Y - Bund 10Y | Calcolato (`IT10Y - DE10Y`) | ECB Data Portal / FRED | Giornaliera | Termometro del rischio sovrano e frammentazione UE |

### Anello 2 (Intermedio): Macroeconomia Reale (Fondamentali)
| ID Nodo | Quadrante | Nome Indicatore | Ticker Ufficiale | Fonte Certificata | Frequenza |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `INF_USCPI` | Inflazione | US CPI All Urban (YoY) | `CPIAUCSL` | FRED / BLS | Mensile |
| `INF_USPCE` | Inflazione | US Core PCE Price Index | `PCEPILFE` | FRED / BEA | Mensile |
| `INF_EXP5Y` | Inflazione | 5Y5Y Forward Inflation Exp. | `T5YIFR` | FRED / Federal Reserve | Giornaliera |
| `INF_EZHICP` | Inflazione | Eurozone HICP Inflation | `CP0000EZ19M086NEST` | Eurostat / ECB / FRED | Mensile |
| `LAB_UNRATE` | Lavoro | US Unemployment Rate (U3) | `UNRATE` | FRED / BLS | Mensile |
| `LAB_NFP` | Lavoro | Non-Farm Payrolls | `PAYEMS` | FRED / BLS | Mensile |
| `LAB_CLAIMS` | Lavoro | Initial Jobless Claims | `ICSA` | FRED / Dept. of Labor | Settimanale |
| `GRO_USGDP` | Crescita | US Real GDP (QoQ Ann.) | `A191RL1Q225SBEA` | FRED / BEA | Trimestrale |
| `GRO_INDPRO` | Crescita | US Industrial Production | `INDPRO` | FRED / Federal Reserve | Mensile |
| `GRO_RETAIL` | Crescita | US Advance Retail Sales | `RSXFS` | FRED / Census Bureau | Mensile |

### Anello 3 (Esterno): Indici Azionari, Credito & Asset Intermarket
| ID Nodo | Categoria | Nome Asset | Ticker Ufficiale | Ruolo Intermarket |
| :--- | :--- | :--- | :--- | :--- |
| `EQ_SPX` | Equity USA | S&P 500 | `^GSPC` | Benchmark azionario globale a larga capitalizzazione |
| `EQ_NDX` | Equity USA | Nasdaq 100 | `^NDX` | Asset a lunga duration tech, ultra-sensibile ai tassi reali |
| `EQ_RUT` | Equity USA | Russell 2000 | `^RUT` | Small-cap domestiche; sensibile a credito bancario e ciclo |
| `EQ_DAX` | Equity UE | DAX 40 | `^GDAXI` | Ciclico industriale europeo ad alta esposizione all'export |
| `EQ_MIB` | Equity UE | FTSE MIB | `FTSEMIB.MI` | Indice bancario; sensibile a tassi BCE e Spread BTP |
| `RISK_VIX` | Rischio | CBOE Volatility Index | `^VIX` | Prezzo delle coperture opzionali istituzionali sull'S&P 500 |
| `CRED_HY` | Credito | ICE BofA US High Yield Spread | `BAMLH0A0HYM2` | Misura dello stress sul credito societario speculativo |
| `FX_DXY` | Valute | US Dollar Index | `DX-Y.NYB` | Condizioni finanziarie globali e liquidità USD mondiale |
| `FX_EURUSD` | Valute | Cambio EUR/USD | `EURUSD=X` | Riflette il differenziale monetario FED/Treasury vs BCE/Bund |
| `COM_GOLD` | Commodities | Oro (Gold Futures) | `GC=F` | Copertura su tassi reali negativi, de-dollarizzazione e fiat |
| `COM_OIL` | Commodities | Petrolio WTI | `CL=F` | Input primario dell'inflazione e tassa implicita sui consumi |
| `COM_COPPER`| Commodities | Rame Future | `HG=F` | "Dr. Copper": anticipatore puro dell'attività industriale |

---

## 🔗 I 10 Fili Maestri della Tela (Cross-Links Econometrici)

| ID Filo | Nodo A ↔ Nodo B | Correlazione Teorica | Meccanismo Macroeconomico & Finanziario | Alert di Rottura / Tensione Critica ($|Z| \ge 2.0$) |
| :---: | :--- | :---: | :--- | :--- |
| **`LINK_01`** | `INF_USPCE` ↔ `YIELD_US02` | **Positiva (+)** | L'inflazione core spinge il mercato a prezzare rialzi (o mancati tagli) della FED sul 2 anni. | Se l'inflazione sale ma il 2Y scende, il mercato teme uno shock recessivo imminente o un pivot forzato. |
| **`LINK_02`** | `YIELD_REAL10` ↔ `EQ_NDX` | **Inversa (-)** | Rendimenti reali più alti riducono il valore attuale dei flussi di cassa futuri delle Big Tech. | Se salgono entrambi, l'azionario è in espansione multipli euforica a rischio di violento repricing. |
| **`LINK_03`** | `YIELD_REAL10` ↔ `COM_GOLD` | **Inversa (-)** | L'oro non paga cedole: soffre la concorrenza quando i Treasury offrono rendimenti reali positivi elevati. | Se salgono entrambi, è in atto accumulazione per rischio debito sovrano o de-dollarizzazione. |
| **`LINK_04`** | `COM_OIL` ↔ `INF_EXP5Y` | **Positiva (+)** | Il prezzo dell'energia trasmette impulsi diretti alle aspettative d'inflazione a medio termine. | Se il petrolio crolla ma le aspettative d'inflazione restano alte, l'inflazione è diventata strutturale nei salari. |
| **`LINK_05`** | `LAB_CLAIMS` ↔ `CURVE_10_2` | **Positiva (+)** | L'aumento dei sussidi disoccupazione forza il bull steepening (il 2Y crolla per attese di tagli FED). | Se i sussidi salgono e la curva si inverte ulteriormente, la FED è percepita 'behind the curve'. |
| **`LINK_06`** | `CRED_HY` ↔ `EQ_RUT` | **Inversa (-)** | L'allargamento degli spread High Yield restringe il credito e soffoca per prime le small-cap del Russell 2000. | Se gli spread di credito salgono mentre gli indici azionari fanno nuovi massimi, il rally azionario è fragile. |
| **`LINK_07`** | `YIELD_US02` ↔ `FX_EURUSD` | **Inversa (-)** | Quando i rendimenti a breve USA salgono più di quelli europei, i flussi sul dollaro deprimono EUR/USD. | Divergenza tra differenziale tassi e cambio segnala stress di liquidità in USD o flussi anomali di rimpatrio. |
| **`LINK_08`** | `FX_DXY` ↔ `EQ_SPX` | **Inversa (-)** | Un dollaro troppo forte drena liquidità globale e riduce gli utili esteri delle multinazionali S&P 500. | Dollaro e S&P 500 entrambi in forte rialzo indicano che gli USA stanno agendo da unico bene rifugio mondiale. |
| **`LINK_09`** | `COM_COPPER` ↔ `GRO_INDPRO` | **Positiva (+)** | Il rame anticipa tempestivamente l'espansione o la contrazione del ciclo manifatturiero reale. | Rame in crollo con produzione industriale ancora forte segnala un imminente rallentamento non ancora nei dati. |
| **`LINK_10`** | `SPREAD_BTP` ↔ `EQ_MIB` | **Inversa (-)** | L'allargamento dello spread sovrano colpisce i bilanci bancari che dominano il FTSE MIB. | Se lo spread sale ma il FTSE MIB ignora il movimento, si accumula un rischio asimmetrico represso sulle banche. |

---

## 🧮 Motore Quantitativo & Formalizzazione Matematica

### 1. Allineamento Serie Temporali a Frequenze Miste (Senza Look-Ahead Bias)
Le serie giornaliere ($D$), settimanali ($W$), mensili ($M$) e trimestrali ($Q$) sono riallineate su un calendario unificato di giorni lavorativi ($B$):
$$X_t = X_{\tau(t)} \quad \text{con } \tau(t) = \max \{ s \in \text{Osservazioni} \mid s \le t \}$$
Nessun dato futuro viene mai utilizzato per interpolare o riallineare le osservazioni passate.

### 2. Z-Score Dinamico di Nodo
Per ogni nodo $i$ su una finestra temporale rolling $W \in \{30D, 90D, 1Y, 5Y\}$:
$$Z_{i,t} = \frac{X_{i,t} - \mu_i(W)}{\sigma_i(W)}$$
- $|Z| < 1.25$: **Normale**
- $1.25 \le |Z| < 2.0$: **Attenzione (Tensione Moderata)**
- $|Z| \ge 2.0$: **Estremo (Anomalia Storica)**

### 3. Tensione del Filo (Z-Score di Relazione)
Dati due nodi $A$ e $B$ legati da una correlazione teorica $s_{AB} \in \{+1, -1\}$, la tensione del filo combina la divergenza normalizzata dei movimenti direzionali e il breakdown della correlazione rolling:
$$Z_{\text{link}} = 0.6 \cdot |D_{\text{dir}}| + 0.8 \cdot \max(0, s_{AB} \cdot (s_{AB} - r_{\text{curr}}))$$
Se due asset storicamente a correlazione inversa iniziano a salire o scendere contemporaneamente, o se la correlazione rolling inverte segno, il filo cambia colore verso l'ambra/viola neon e segnala una divergenza critica.

### 4. Systemic Web Tension (Indice di Tensione della Tela)
Media ponderata degli Z-score di tensione di tutti i fili maestri:
$$\text{Web Tension} = \frac{\sum_{k=1}^{10} w_k |Z_{\text{link}, k}|}{\sum_{k=1}^{10} w_k}$$
Se $\text{Web Tension} \ge 2.0$, il Centro di Controllo lancia l'allarme di rottura sistemica dei regimi di mercato.

### 5. Regola di Taylor Istituzionale & Policy Bias
$$r_{\text{Taylor}} = r^* + \pi_{\text{PCE}} + 0.5(\pi_{\text{PCE}} - \pi^*) + 0.5(y_{\text{GDP}} - y^*)$$
Con $r^* = 0.5\%$, $\pi^* = 2.0\%$, $y^* = 2.0\%$.

---

## 🚀 Come Avviare l'Applicazione

### Metodo 1: Script Automatico Diretto
```bash
bash run.sh
```

### Metodo 2: Esecuzione Streamlit
```bash
python3 -m streamlit run app.py --server.port 8502
```

L'applicazione sarà accessibile all'indirizzo:  
👉 **`http://localhost:8502`** (oppure da smartphone/tablet via Wi-Fi tramite l'indirizzo locale mostrato all'avvio).

---

## 🧪 Esecuzione dei Test Unitari
Per verificare la correttezza di tutta l'architettura quantitativa, dei ticker istituzionali e dei generatori di grafici:
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

---

## 📂 Struttura del Software

```text
SPIDER/
├── app.py                  # Entry point Streamlit: interfaccia, sidebar, KPI e navigazione a schede
├── config.py               # Ticker istituzionali, coordinate polari 30 nodi, 10 fili maestri e soglie Z
├── data_fetcher.py         # Connettori istituzionali (FRED, BCE, Yahoo, BLS) con cache SQLite
├── quant_engine.py         # Motore econometrico: allineamento, correlazioni rolling, Z-score, Taylor rule
├── spider_ui.py            # Grafo Plotly concentrico a tela di ragno, matrice 2D e generatori tabelle
├── run.sh                  # Script bash di avvio con verifica dipendenze e IP Wi-Fi
├── requirements.txt        # Dipendenze Python
├── README.md               # Documentazione scientifica, formule e guida utente
├── data/
│   └── spider_cache.db     # Database SQLite persistente delle serie storiche e metadati
└── tests/
    └── test_spider.py      # Test unitari automatizzati del motore quantitativo e UI
```

---
*© 2026 SPIDER • Macro & Market Network Monitor. Progettato con standard quantitativi e matematici istituzionali.*
