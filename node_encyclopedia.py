"""
SPIDER Node Encyclopedia Module:
Comprehensive Institutional Documentation for all 30 Macro-Financial Nodes.
Structured with: Definizione, Cosa misura esattamente, Interpretazione, Perché è rilevante, Fonti e ticker.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class NodeManualEntry:
    id: str
    name: str
    ring: int
    quadrant: str
    definizione: str
    cosa_misura: str
    interpretazione: str
    perche_rilevante: str
    fonti_ticker: str


NODE_ENCYCLOPEDIA: Dict[str, NodeManualEntry] = {
    # =========================================================================
    # ANELLO 1 (INTERNO): POLITICA MONETARIA, TASSI & RENDIMENTI (8 NODI)
    # =========================================================================
    "RATE_FED": NodeManualEntry(
        id="RATE_FED",
        name="Fed Funds Effective Rate",
        ring=1,
        quadrant="LIQUIDITY_RISK",
        definizione="Il tasso effettivo sui fondi federali è il tasso di interesse al quale le banche e gli istituti di credito depositari negli Stati Uniti si prestano riserve di liquidità non garantite (uncollateralized) su base overnight presso la Federal Reserve.",
        cosa_misura="Il costo medio ponderato per volume delle transazioni overnight sul mercato interbancario delle riserve della Federal Reserve (in percentuale annua %).",
        interpretazione="Se il tasso sale, la banca centrale sta attuando una stretta creditizia drenando liquidità; se scende, sta allentando le condizioni finanziarie. Un tasso superiore al tasso neutrale di lungo termine (r* + target d'inflazione) deprime la domanda aggregata e raffredda l'economia.",
        perche_rilevante="È il prezzo base del capitale a livello mondiale: da esso derivano direttamente o indirettamente i tassi sui mutui, i prestiti corporate, i tassi repo e il tasso privo di rischio (risk-free rate) globale.",
        fonti_ticker="Ticker Ufficiale FRED: DFF | Fonte: Federal Reserve Bank of New York / FRED St. Louis."
    ),

    "RATE_ECB": NodeManualEntry(
        id="RATE_ECB",
        name="ECB Deposit Facility Rate",
        ring=1,
        quadrant="LIQUIDITY_RISK",
        definizione="Il tasso sui depositi presso la Banca Centrale Europea definisce l'interesse che le banche commerciali dell'Eurozona ricevono per depositare la propria liquidità eccedente overnight presso l'Eurosistema.",
        cosa_misura="Il rendimento garantito sui saldi overnight detenuti dalle banche dell'Eurosistema oltre la riserva obbligatoria (in percentuale annua %).",
        interpretazione="Con l'abbondanza strutturale di riserve post-QE, il DFR agisce come floor operativo primario dell'Euribor e del tasso €STR. Un DFR elevato frena l'offerta di credito bancario europeo e difende la stabilità dei prezzi, mentre tassi bassi o negativi stimolano l'erogazione creditizia.",
        perche_rilevante="È l'ancora monetaria fondamentale per l'economia dell'Eurozona; determina il costo di rifinanziamento per governi, imprese e famiglie nei 20 paesi membri.",
        fonti_ticker="Ticker Ufficiale FRED: ECBDFR | Fonte: Banca Centrale Europea (ECB) / FRED."
    ),

    "YIELD_US02": NodeManualEntry(
        id="YIELD_US02",
        name="US Treasury 2Y Yield",
        ring=1,
        quadrant="LIQUIDITY_RISK",
        definizione="Il rendimento a scadenza (Yield to Maturity) dell'obbligazione benchmark a 2 anni emessa dal Dipartimento del Tesoro degli Stati Uniti.",
        cosa_misura="Il rendimento annuo effettivo a scadenza dei titoli sovrani statunitensi a 24 mesi scambiati sul mercato secondario (in %).",
        interpretazione="Incorpora quasi esclusivamente le aspettative di mercato sulla traiettoria dei Fed Funds nei successivi 12-24 mesi più un piccolo premio per il termine. Se il rendimento a 2 anni scende nettamente sotto il Fed Funds rate corrente, il mercato sconta tagli imminenti dei tassi; se sale sopra, sconta ulteriori rialzi.",
        perche_rilevante="È il termometro più tempestivo e affidabile per anticipare il punto di svolta (pivot) della Federal Reserve ed è il driver primario dei tassi bancari a breve e del valore del dollaro USA.",
        fonti_ticker="Ticker Ufficiale FRED: DGS2 | Fonte: U.S. Department of the Treasury / FRED."
    ),

    "YIELD_US10": NodeManualEntry(
        id="YIELD_US10",
        name="US Treasury 10Y Yield",
        ring=1,
        quadrant="LIQUIDITY_RISK",
        definizione="Il rendimento a scadenza del titolo governativo benchmark a 10 anni emesso dal Tesoro statunitense, considerato l'asset privo di rischio di credito per eccellenza a medio-lungo termine.",
        cosa_misura="Il rendimento nominale annuo del debito sovrano USA a scadenza 10 anni negoziato sul mercato obbligazionario globale (in %).",
        interpretazione="Sintetizza tre componenti: aspettative di crescita economica reale, aspettative d'inflazione media decennale e il term premium (premio per il rischio di durata). Se sale per aspettative di crescita favorisce i ciclici; se sale per shock d'inflazione o deficit fiscale comprime i multipli azionari.",
        perche_rilevante="È il tasso di sconto (discount rate) universale applicato nei modelli di Discounted Cash Flow (DCF) per valutare l'azionario globale, oltre che il benchmark per i tassi sui mutui a tasso fisso USA a 30 anni.",
        fonti_ticker="Ticker Ufficiale FRED: DGS10 | Yahoo Finance: ^TNX | Fonte: U.S. Department of the Treasury / CME / FRED."
    ),

    "YIELD_REAL10": NodeManualEntry(
        id="YIELD_REAL10",
        name="US 10Y TIPS (Real Yield)",
        ring=1,
        quadrant="INFLATION",
        definizione="Il rendimento reale offerto dai Treasury Inflation-Protected Securities (TIPS) a 10 anni, ovvero il rendimento sovrano depurato contrattualmente dall'inflazione attesa.",
        cosa_misura="Il rendimento reale annuo netto garantito sopra l'indice dell'inflazione CPI su un orizzonte decennale (in %).",
        interpretazione="Esprime la reale rigidità delle condizioni monetarie e il costo reale del capitale. Rendimenti reali positivi ed elevati (>1.5%-2.0%) creano un forte vento contrario per gli asset a lunga duration (Tech/Growth) e aumentano il costo opportunità di detenere oro fisico. Rendimenti reali negativi innescano rally su oro e asset rifugio.",
        perche_rilevante="È la variabile finanziaria più influente per le valutazioni azionarie del settore tecnologico (Nasdaq) e per il prezzo dell'oro, nonché la misura empirica del tasso neutrale reale di lungo periodo.",
        fonti_ticker="Ticker Ufficiale FRED: DFII10 | Fonte: U.S. Department of the Treasury / FRED."
    ),

    "CURVE_10_2": NodeManualEntry(
        id="CURVE_10_2",
        name="Spread Curva USA 10Y-2Y",
        ring=1,
        quadrant="GROWTH",
        definizione="Il differenziale di rendimento tra il titolo di stato statunitense a 10 anni e quello a 2 anni (YIELD_US10 meno YIELD_US02).",
        cosa_misura="La pendenza (slope) della curva dei rendimenti governativi americani sul tratto intermedio della curva (in punti base / bps).",
        interpretazione="Un valore negativo (curva invertita) è il più affidabile segnale predittivo di recessione a 12-18 mesi. Quando la curva si disinverte rialzandosi per crollo dei tassi a breve (Bull Steepening), la recessione è imminente; se si disinverte per rialzo dei tassi lunghi (Bear Steepening), il mercato sconta espansione inflazionistica (No Landing).",
        perche_rilevante="Regola il margine di interesse e la capacità di erogazione del credito del sistema bancario commerciale (trasformazione delle scadenze) e scandisce le fasi del ciclo intermarket.",
        fonti_ticker="Ticker Ufficiale FRED: T10Y2Y | Fonte: Federal Reserve Bank of St. Louis."
    ),

    "YIELD_DE10": NodeManualEntry(
        id="YIELD_DE10",
        name="Bund Tedesco 10Y",
        ring=1,
        quadrant="LIQUIDITY_RISK",
        definizione="Il rendimento a scadenza dell'obbligazione benchmark a 10 anni emessa dalla Repubblica Federale di Germania (Bund).",
        cosa_misura="Il tasso di rendimento annuo del titolo sovrano privo di rischio di credito denominato in Euro a 10 anni (in %).",
        interpretazione="Riflette lo stato di salute fondamentale, il potenziale di crescita economica e le aspettative d'inflazione dell'intera Eurozona. Un Bund in ascesa denota espansione economica o pressioni sui salari; un calo marcato riflette timori di recessione o fuga verso la sicurezza (flight to safety).",
        perche_rilevante="Costituisce il benchmark privo di rischio dell'Eurozona, sul quale vengono prezzati tutti gli spread sovrani dei paesi membri (BTP, Bonos, OAT) e il debito corporate europeo.",
        fonti_ticker="Ticker Ufficiale FRED: IRLTLT01DEM156N | Fonte: Deutsche Bundesbank / Banca Centrale Europea / FRED."
    ),

    "SPREAD_BTP": NodeManualEntry(
        id="SPREAD_BTP",
        name="Spread BTP 10Y - Bund 10Y",
        ring=1,
        quadrant="LIQUIDITY_RISK",
        definizione="Il differenziale di rendimento tra il Buono del Tesoro Poliennale (BTP) italiano decennale e il corrispettivo Bund tedesco a 10 anni.",
        cosa_misura="Il premio per il rischio di credito sovrano, di liquidità e di ridenominazione richiesto dal mercato per detenere debito italiano rispetto al debito tedesco (in punti base / bps).",
        interpretazione="Spread compressi (<120 bps) riflettono fiducia nella convergenza fiscale, appetito per il rendimento (carry trade) e credibilità del supporto BCE (scudo TPI). Uno spread in allargamento rapido (>180-200 bps) segnala stress da frammentazione dell'Eurozona, tensioni sul bilancio pubblico e deflussi di capitale.",
        perche_rilevante="È l'indicatore principe di rischio sistemico nell'Eurozona; impatta istantaneamente il patrimonio e la redditività delle banche italiane e il costo del credito alle imprese nazionali.",
        fonti_ticker="Ticker Ufficiale: SPREAD_IT_DE | Fonte: ECB Data Portal / TradingView / FRED."
    ),

    # =========================================================================
    # ANELLO 2 (INTERMEDIO): MACROECONOMIA REALE (FONDAMENTALI) (10 NODI)
    # =========================================================================
    "INF_USCPI": NodeManualEntry(
        id="INF_USCPI",
        name="US CPI All Urban (YoY)",
        ring=2,
        quadrant="INFLATION",
        definizione="L'Indice dei Prezzi al Consumo per tutti i consumatori urbani negli Stati Uniti (Consumer Price Index - Headline), calcolato dal Bureau of Labor Statistics.",
        cosa_misura="La variazione percentuale tendenziale annua (%) del costo di un paniere fisso rappresentativo di beni e servizi acquistati dalle famiglie urbane americane, comprensivo di alimentari ed energia.",
        interpretazione="Misura l'inflazione diretta percepita dal consumatore e dai lavoratori. Un valore persistentemente sopra il 2% erode il reddito reale disponibile, comprime i margini aziendali non protetti da pricing power e costringe le banche centrali ad alzare i tassi.",
        perche_rilevante="È il dato sull'inflazione più seguito e mediaticamente influente al mondo; condiziona i rinnovi contrattuali salariali e le aspettative inflazionistiche a breve termine.",
        fonti_ticker="Ticker Ufficiale FRED: CPIAUCSL | Fonte: U.S. Bureau of Labor Statistics (BLS) / FRED."
    ),

    "INF_USPCE": NodeManualEntry(
        id="INF_USPCE",
        name="US Core PCE Price Index",
        ring=2,
        quadrant="INFLATION",
        definizione="L'indice dei prezzi delle spese per consumi personali (Personal Consumption Expenditures) depurato delle componenti volatili dei beni alimentari e dell'energia (Core PCE).",
        cosa_misura="La variazione percentuale tendenziale annua (%) dei prezzi effettivamente pagati dai consumatori per tutti i consumi individuali, con ponderazioni dinamiche che tengono conto della sostituzione dei beni da parte delle famiglie.",
        interpretazione="Rappresenta l'indicatore ufficiale di stabilità dei prezzi statutariamente perseguito dalla Federal Reserve per il target simmetrico del 2.0%. Valori sopra il 2.5% segnalano inflazione radicata nei servizi e impongono tassi reali restrittivi.",
        perche_rilevante="È la misura regina di inflazione inserita nella Regola di Taylor per quantificare il tasso d'interesse neutrale e prescritto alla Federal Reserve.",
        fonti_ticker="Ticker Ufficiale FRED: PCEPILFE | Fonte: U.S. Bureau of Economic Analysis (BEA) / FRED."
    ),

    "INF_EXP5Y": NodeManualEntry(
        id="INF_EXP5Y",
        name="5Y5Y Forward Inflation Exp.",
        ring=2,
        quadrant="INFLATION",
        definizione="Il tasso di inflazione medio a 5 anni atteso dal mercato obbligazionario a partire da 5 anni nel futuro (segmento 5Y-10Y forward).",
        cosa_misura="Il tasso annuo atteso di inflazione a medio-lungo termine implicito nei prezzi degli Inflation Swaps e nella curva dei titoli TIPS (in %).",
        interpretazione="Misura il grado di ancoraggio delle aspettative inflazionistiche a lungo termine. Valori stabili tra 2.0% e 2.3% certificano la credibilità della politica monetaria; superamenti del 2.5% denotano perdita di fiducia nella capacità della banca centrale di dominare l'inflazione.",
        perche_rilevante="È l'indicatore prediletto dai comitati di politica monetaria (FOMC) per distinguere fiammate transitorie delle materie prime da pericoli di de-ancoraggio strutturale.",
        fonti_ticker="Ticker Ufficiale FRED: T5YIFR | Fonte: Federal Reserve Bank of St. Louis."
    ),

    "INF_EZHICP": NodeManualEntry(
        id="INF_EZHICP",
        name="Eurozone HICP Inflation",
        ring=2,
        quadrant="INFLATION",
        definizione="L'Indice Armonizzato dei Prezzi al Consumo (Harmonised Index of Consumer Prices) aggregato dell'area Euro, calcolato con metodologia unificata da Eurostat.",
        cosa_misura="Il tasso tendenziale annuo di variazione dei prezzi al consumo (%) ponderato sul paniere dei 20 paesi aderenti alla moneta unica europea.",
        interpretazione="Un HICP superiore al target statutario del 2.0% costringe la BCE a prolungare politiche restrittive o alzare il Deposit Facility Rate. Una discesa rapida verso il target crea spazio per allentamenti monetari e supporto ai titoli periferici.",
        perche_rilevante="È il target statutario esclusivo della Banca Centrale Europea; ogni decisione su tassi, QE o QT nell'Eurozona dipende dalla traiettoria di questa metrica.",
        fonti_ticker="Ticker Ufficiale FRED: CP0000EZ19M086NEST | Fonte: Eurostat / Banca Centrale Europea / FRED."
    ),

    "LAB_UNRATE": NodeManualEntry(
        id="LAB_UNRATE",
        name="US Unemployment Rate (U3)",
        ring=2,
        quadrant="LABOR",
        definizione="Il tasso di disoccupazione ufficiale degli Stati Uniti (U3), calcolato mensilmente dal Bureau of Labor Statistics tramite l'indagine campionaria sulle famiglie (Household Survey).",
        cosa_misura="La percentuale (%) della forza lavoro civile americana che non possiede un impiego, è disponibile a lavorare ed è stata attivamente alla ricerca di lavoro nelle quattro settimane precedenti.",
        interpretazione="Tassi ai minimi storici indicano mercato del lavoro rigido (tight labor market), con rischi di spirale salari-prezzi. La 'Sahm Rule' segnala una recessione certa non appena la media mobile a 3 mesi del tasso sale di almeno 0.50 punti percentuali rispetto al minimo degli ultimi 12 mesi.",
        perche_rilevante="Rappresenta uno dei due pilastri del doppio mandato della Federal Reserve (massima occupazione e stabilità dei prezzi) e funge da barometro sociale del ciclo economico.",
        fonti_ticker="Ticker Ufficiale FRED: UNRATE | Fonte: U.S. Bureau of Labor Statistics (BLS) / FRED."
    ),

    "LAB_NFP": NodeManualEntry(
        id="LAB_NFP",
        name="Non-Farm Payrolls (Level)",
        ring=2,
        quadrant="LABOR",
        definizione="Il numero totale di lavoratori dipendenti occupati nel settore non agricolo statunitense, rilevato mensilmente dalla Establishment Survey presso oltre 600.000 aziende.",
        cosa_misura="La variazione netta mensile e il livello complessivo delle buste paga nel settore privato e pubblico USA, escluso il comparto agricolo (espresso in migliaia di dipendenti).",
        interpretazione="Una creazione superiore a 150-200 mila posti di lavoro al mese segnala espansione solida; valori inferiori a 100 mila denotano rallentamento ciclico, mentre valori negativi indicano contrazione economica in corso.",
        perche_rilevante="È la statistica macroeconomica mensile a più alto impatto sui mercati: le sorprese rispetto al consensus muovono violentemente tassi, valute e borse globali.",
        fonti_ticker="Ticker Ufficiale FRED: PAYEMS | Fonte: U.S. Bureau of Labor Statistics (BLS) / FRED."
    ),

    "LAB_CLAIMS": NodeManualEntry(
        id="LAB_CLAIMS",
        name="Initial Jobless Claims",
        ring=2,
        quadrant="LABOR",
        definizione="Il numero settimanale di cittadini statunitensi che presentano per la prima volta richiesta di indennità di disoccupazione presso gli uffici del lavoro statali.",
        cosa_misura="Il flusso settimanale lordo di nuovi licenziamenti nell'economia statunitense (espresso in numero assoluto di domande).",
        interpretazione="Essendo rilasciato con cadenza settimanale ogni giovedì, è l'indicatore macro ad alta frequenza più rapido in assoluto. Valori stabilmente sopra 250k-300k indicano contrazione del mercato del lavoro; valori inferiori a 220k confermano assenza di licenziamenti diffusi.",
        perche_rilevante="È un leading indicator affidabile del mercato del lavoro: anticipa con settimane di anticipo le svolte nei dati mensili di disoccupazione e occupazione.",
        fonti_ticker="Ticker Ufficiale FRED: ICSA | Fonte: U.S. Department of Labor / FRED."
    ),

    "GRO_USGDP": NodeManualEntry(
        id="GRO_USGDP",
        name="US Real GDP (QoQ Ann.)",
        ring=2,
        quadrant="GROWTH",
        definizione="Il Prodotto Interno Lordo reale statunitense, rettificato per la variazione del livello generale dei prezzi (inflazione) dal Bureau of Economic Analysis.",
        cosa_misura="Il tasso di crescita percentuale trimestrale annualizzato (%) del valore monetario di tutti i beni e servizi finali prodotti all'interno degli Stati Uniti.",
        interpretazione="Crescite sopra il potenziale (>2.0%-2.5%) riflettono robustezza o surriscaldamento; crescite tra 1.0% e 2.0% indicano atterraggio morbido (Soft Landing); valori negativi prolungati certificano recessione economica.",
        perche_rilevante="È la metrica definitiva della produzione aggregata di una nazione; quantifica l'Output Gap (yt - y*) all'interno della Regola di Taylor.",
        fonti_ticker="Ticker Ufficiale FRED: A191RL1Q225SBEA | Fonte: U.S. Bureau of Economic Analysis (BEA) / FRED."
    ),

    "GRO_INDPRO": NodeManualEntry(
        id="GRO_INDPRO",
        name="US Industrial Production (YoY)",
        ring=2,
        quadrant="GROWTH",
        definizione="L'indice di produzione industriale calcolato mensilmente dal Federal Reserve Board per monitorare l'output fisico del settore industriale.",
        cosa_misura="La variazione percentuale tendenziale annua (%) dell'output reale generato dagli impianti manifatturieri, minerari e dei servizi di pubblica utilità (gas ed elettricità) negli USA.",
        interpretazione="Misura l'attività dal lato dell'offerta e dei beni materiali durevoli. È altamente prociclico ed è strettamente correlato con la domanda di materie prime industriali (rame, greggio) e con gli investimenti fissi delle aziende.",
        perche_rilevante="È una delle 4 variabili coincidenti primarie utilizzate dal NBER (National Bureau of Economic Research) per la datazione ufficiale dei cicli di recessione ed espansione negli Stati Uniti.",
        fonti_ticker="Ticker Ufficiale FRED: INDPRO | Fonte: Federal Reserve Board / FRED."
    ),

    "GRO_RETAIL": NodeManualEntry(
        id="GRO_RETAIL",
        name="US Retail Sales (YoY)",
        ring=2,
        quadrant="GROWTH",
        definizione="Il valore totale stimato delle vendite di beni al dettaglio e servizi di ristorazione negli Stati Uniti, monitorato mensilmente dallo U.S. Census Bureau.",
        cosa_misura="La variazione percentuale tendenziale annua (%) della spesa monetaria dei consumatori presso i canali di vendita al dettaglio, escluso il comparto auto (RSXFS) per depurare l'indice da oscillazioni di fornitura temporanee.",
        interpretazione="Poiché i consumi privati costituiscono circa il 68-70% del PIL americano, vendite al dettaglio positive e resilienti confermano la capacità di spesa delle famiglie e supportano scenari di Soft o No Landing. Un crollo delle vendite preannuncia recessione.",
        perche_rilevante="Funge da termometro diretto della domanda interna finale e della propensione al consumo del cittadino americano.",
        fonti_ticker="Ticker Ufficiale FRED: RSXFS | Fonte: U.S. Census Bureau / FRED."
    ),

    # =========================================================================
    # ANELLO 3 (ESTERNO): INDICI AZIONARI, CREDITO & ASSET INTERMARKET (12 NODI)
    # =========================================================================
    "EQ_SPX": NodeManualEntry(
        id="EQ_SPX",
        name="S&P 500",
        ring=3,
        quadrant="GROWTH",
        definizione="L'indice azionario ponderato per la capitalizzazione di mercato aggiustata per il flottante delle 500 principali società quotate sulle borse statunitensi (NYSE e NASDAQ).",
        cosa_misura="Il livello di prezzo e la performance complessiva dell'azionario statunitense a grande capitalizzazione (large-cap) in punti indice.",
        interpretazione="Riflette la fiducia degli investitori, le aspettative di crescita degli utili aziendali e la propensione al rischio istituzionale. Massimi storici indicano condizioni finanziarie permissive o aspettative di forte crescita economica.",
        perche_rilevante="È il benchmark azionario di riferimento per l'intero pianeta finanziario; a esso sono legati trilioni di dollari di fondi indicizzati, derivati e piani pensionistici globali.",
        fonti_ticker="Ticker Ufficiale: ^GSPC | Fonte: S&P Dow Jones Indices / CME / Yahoo Finance."
    ),

    "EQ_NDX": NodeManualEntry(
        id="EQ_NDX",
        name="Nasdaq 100",
        ring=3,
        quadrant="GROWTH",
        definizione="L'indice azionario che raggruppa le 100 maggiori società non finanziarie quotate sul mercato elettronico Nasdaq.",
        cosa_misura="Il valore aggregato delle aziende leader mondiali nei settori tecnologia, intelligenza artificiale, biotecnologie, software e commercio digitale (in punti).",
        interpretazione="Essendo composto da società con cash flow previsti prevalentemente nel futuro (lunga duration finanziaria), è ipersensibile alle variazioni dei rendimenti reali dei TIPS (YIELD_REAL10). Tassi reali in calo espandono i multipli (P/E); tassi reali in salita comprimono le valutazioni.",
        perche_rilevante="È il motore dell'innovazione tecnologica globale e funge da catalizzatore dei rally azionari contemporanei basati su intelligenza artificiale e produttività.",
        fonti_ticker="Ticker Ufficiale: ^NDX | Fonte: Nasdaq, Inc. / Yahoo Finance."
    ),

    "EQ_RUT": NodeManualEntry(
        id="EQ_RUT",
        name="Russell 2000",
        ring=3,
        quadrant="GROWTH",
        definizione="L'indice azionario che replica la performance delle 2.000 aziende a minore capitalizzazione all'interno dell'universo del Russell 3000 Index.",
        cosa_misura="Il prezzo e la performance aggregata delle piccole e medie imprese domestiche quotate negli Stati Uniti (small-cap) in punti indice.",
        interpretazione="Le small-cap americane presentano mediamente debiti a tasso variabile più elevati e dipendono dal canale bancario tradizionale. Se il Russell sovraperforma l'S&P 500, la crescita economica è diffusa e democratica; se sottoperforma, la stretta creditizia sta colpendo l'anello debole del tessuto economico.",
        perche_rilevante="È il barometro più puro della salute dell'economia domestica USA e della trasmissione del credito bancario all'economia reale.",
        fonti_ticker="Ticker Ufficiale: ^RUT | Fonte: FTSE Russell / Yahoo Finance."
    ),

    "EQ_DAX": NodeManualEntry(
        id="EQ_DAX",
        name="DAX 40",
        ring=3,
        quadrant="GROWTH",
        definizione="L'indice benchmark della borsa di Francoforte, composto dalle 40 maggiori aziende per capitalizzazione e liquidità del mercato azionario tedesco.",
        cosa_misura="La performance complessiva con dividendi reinvestiti (Performanceindex) dei campioni industriali, chimici, automobilistici e farmaceutici della Germania (in punti).",
        interpretazione="Fortemente orientato alla manifattura pesante e all'export globale (particolarmente verso Cina e USA). Un DAX forte segnala ripresa del commercio mondiale e della produzione industriale europea; debolezza del DAX segnala shock energetici o de-industrializzazione continentale.",
        perche_rilevante="Rappresenta l'ancora azionaria della maggiore economia dell'Eurozona e riflette lo stato di salute del commercio internazionale manifatturiero.",
        fonti_ticker="Ticker Ufficiale: ^GDAXI | Fonte: Deutsche Börse / Yahoo Finance."
    ),

    "EQ_MIB": NodeManualEntry(
        id="EQ_MIB",
        name="FTSE MIB",
        ring=3,
        quadrant="LIQUIDITY_RISK",
        definizione="Il principale indice di riferimento del mercato azionario italiano di Piazza Affari, composto dalle 40 società più scambiate e capitalizzate.",
        cosa_misura="La performance ponderata delle blue-chip quotate su Borsa Italiana a Milano (in punti base).",
        interpretazione="Con una concentrazione molto alta di banche e assicurazioni (>30-35%), il FTSE MIB è direttamente correlato al margine di interesse bancario, alla politica della BCE e allo spread BTP/Bund. Un allargamento dello spread sovrano deprime istantaneamente le banche nel MIB.",
        perche_rilevante="Funge da termometro dell'Europa meridionale e del settore creditizio europeo periferico; riflette la sensibilità al rischio debito sovrano.",
        fonti_ticker="Ticker Ufficiale: FTSEMIB.MI | Fonte: Borsa Italiana / Euronext / Yahoo Finance."
    ),

    "RISK_VIX": NodeManualEntry(
        id="RISK_VIX",
        name="CBOE Volatility Index",
        ring=3,
        quadrant="LIQUIDITY_RISK",
        definizione="L'indice di volatilità implicita calcolato dal Chicago Board Options Exchange (CBOE), noto sui mercati come 'Indice della Paura'.",
        cosa_misura="La volatilità annualizzata attesa a 30 giorni dell'indice S&P 500, estrapolata dai prezzi delle opzioni put e call out-of-the-money (in punti percentuali).",
        interpretazione="Valori bassi (<15) indicano compiacenza e bassa percezione del rischio; valori tra 15 e 25 riflettono normale volatilità ciclica; picchi sopra 30-40 indicano panico, liquidazione forzata delle posizioni a leva (deleveraging) e corsa alle coperture.",
        perche_rilevante="Regola automaticamente la quantità di rischio assunta dai fondi sistematici istituzionali (Volatility Target, Risk Parity); un'impennata del VIX causa vendite forzate automatiche di azionario.",
        fonti_ticker="Ticker Ufficiale: ^VIX | Fonte: Chicago Board Options Exchange (CBOE) / Yahoo Finance."
    ),

    "CRED_HY": NodeManualEntry(
        id="CRED_HY",
        name="ICE BofA US High Yield Spread",
        ring=3,
        quadrant="LIQUIDITY_RISK",
        definizione="Lo spread Option-Adjusted (OAS) calcolato dall'indice ICE BofA sulle obbligazioni societarie statunitensi con rating speculativo (sub-investment grade, inferiori a BBB-).",
        cosa_misura="Il differenziale di rendimento (in punti base o %) che le aziende private ad alto rischio di default devono corrispondere rispetto ai titoli governativi del Tesoro USA a pari scadenza.",
        interpretazione="Spread compressi (<350 bps) segnalano liquidità abbondante e fiducia nella solvibilità societaria. Un'impennata degli spread (>500-600 bps) indica credit crunch, stress di rifinanziamento per le aziende fragili e imminente ondata di fallimenti.",
        perche_rilevante="È il canarino nella miniera dei mercati: le correzioni azionarie violente e le recessioni sono quasi sempre precedute dall'allargamento preventivo degli spread High Yield.",
        fonti_ticker="Ticker Ufficiale FRED: BAMLH0A0HYM2 | Fonte: ICE Data Indices, LLC / FRED."
    ),

    "FX_DXY": NodeManualEntry(
        id="FX_DXY",
        name="US Dollar Index",
        ring=3,
        quadrant="LIQUIDITY_RISK",
        definizione="L'indice geometrico ponderato che misura il valore del dollaro statunitense rispetto a un paniere di 6 principali valute mondiali (Euro 57.6%, Yen 13.6%, Sterlina 11.9%, Dollaro Canadese 9.1%, Corona Svedese 4.2%, Franco Svizzero 3.6%).",
        cosa_misura="La forza o debolezza relativa del dollaro USA sul mercato dei cambi internazionale (in punti indice con base 100).",
        interpretazione="Un dollaro forte drena liquidità globale, appesantisce i paesi emergenti indebitati in dollari e costituisce un vento contrario per le materie prime. Un dollaro debole agisce come stimolo monetario globale, allentando le condizioni finanziarie mondiali.",
        perche_rilevante="Essendo il dollaro la valuta di riserva e di fatturazione del commercio e del debito internazionale, il DXY è il pivot della liquidità globale.",
        fonti_ticker="Ticker Ufficiale: DX-Y.NYB | Fonte: Intercontinental Exchange (ICE) / Yahoo Finance."
    ),

    "FX_EURUSD": NodeManualEntry(
        id="FX_EURUSD",
        name="Cambio EUR/USD",
        ring=3,
        quadrant="LIQUIDITY_RISK",
        definizione="Il tasso di cambio spot sul mercato valutario (Forex) tra la moneta unica europea (Euro) e il Dollaro statunitense.",
        cosa_misura="La quantità di dollari USA necessari per acquistare un euro (in USD/EUR).",
        interpretazione="Riflette il differenziale nei rendimenti a breve termine tra FED e BCE (LINK_01), i differenziali di crescita economica e i flussi della bilancia dei pagamenti. Un euro forte riduce i costi dell'energia importata ma frena l'export europeo; un euro debole stimola le esportazioni ma importa inflazione energetica.",
        perche_rilevante="È la coppia di valute più scambiata al mondo; determina i flussi commerciali globali e le condizioni finanziarie relative tra le due principali economie occidentali.",
        fonti_ticker="Ticker Ufficiale: EURUSD=X | Fonte: European Central Bank / Forex Market / Yahoo Finance."
    ),

    "COM_GOLD": NodeManualEntry(
        id="COM_GOLD",
        name="Oro (Gold Futures)",
        ring=3,
        quadrant="INFLATION",
        definizione="Il prezzo del contratto future di riferimento sul metallo prezioso fisico (100 once troy di oro fino) scambiato sulla divisione COMEX del CME.",
        cosa_misura="La quotazione di mercato dell'oncia troy d'oro espressa in dollari statunitensi ($/oz).",
        interpretazione="L'oro è un asset senza rischio di controparte che non stacca cedole. Tende ad apprezzarsi fortemente quando i rendimenti reali dei bond governativi scendono o diventano negativi, quando le banche centrali espandono la moneta fiat, o in contesti di de-dollarizzazione e tensione geopolitica.",
        perche_rilevante="È la riserva monetaria suprema neutrale delle banche centrali mondiali e la più collaudata copertura storica contro la perdita di potere d'acquisto delle valute cartacee.",
        fonti_ticker="Ticker Ufficiale: GC=F | Fonte: COMEX / CME Group / Yahoo Finance."
    ),

    "COM_OIL": NodeManualEntry(
        id="COM_OIL",
        name="Petrolio WTI",
        ring=3,
        quadrant="INFLATION",
        definizione="Il prezzo del contratto future di riferimento sul greggio leggero e dolce West Texas Intermediate (WTI) scambiato sul circuito NYMEX.",
        cosa_misura="Il prezzo di mercato di un barile (circa 159 litri) di petrolio greggio statunitense con consegna fisica a Cushing, Oklahoma (in $/bbl).",
        interpretazione="Il petrolio è il costo energetico primario della civiltà industriale e dei trasporti. Un rialzo violento genera inflazione da offerta (cost-push) e agisce come una tassa regressiva sul reddito dei consumatori, deprimendo la crescita economica.",
        perche_rilevante="È la materia prima più scambiata al mondo; guida direttamente le aspettative d'inflazione a breve termine (LINK_04) e determina i surplus/deficit dei paesi produttori ed esportatori.",
        fonti_ticker="Ticker Ufficiale: CL=F | Fonte: NYMEX / CME Group / Yahoo Finance."
    ),

    "COM_COPPER": NodeManualEntry(
        id="COM_COPPER",
        name="Rame (Copper Futures)",
        ring=3,
        quadrant="GROWTH",
        definizione="Il contratto future sul rame di grado A scambiato sulla divisione COMEX del CME Group.",
        cosa_misura="Il prezzo all'ingrosso di una libbra di rame raffinato in dollari statunitensi ($/lb).",
        interpretazione="Soprannominato nei mercati 'Dr. Copper' per via della sua capacità di diagnosticare la salute economica mondiale prima di qualsiasi statistica ufficiale. Essendo impiegato capillarmente in edilizia, reti elettriche, automobili e componenti industriali, un rally del rame anticipa riaccelerazioni dell'attività industriale (LINK_09).",
        perche_rilevante="È il leading indicator ciclico delle materie prime per eccellenza ed è il metallo strategico fondamentale per l'elettrificazione, l'intelligenza artificiale e la transizione energetica globale.",
        fonti_ticker="Ticker Ufficiale: HG=F | Fonte: COMEX / CME Group / Yahoo Finance."
    ),
}
