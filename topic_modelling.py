"""
Topic modelling and comparison of three Serbian (Latin script) corpora:
pink.txt, rts.txt, tanjug.txt

Model: BERTopic (multilingual sentence embeddings + UMAP + HDBSCAN + c-TF-IDF)

Why BERTopic for Serbian?
  - Serbian is highly inflected. LDA/NMF on bag-of-words needs good lemmatisation/stemming
    to work well, and there is no lightweight, reliable lemmatiser for Serbian Latin.
  - Multilingual transformer embeddings (XLM-R family) know Serbian, so semantically
    similar sentences cluster together even when the surface forms differ.
  - That lets us keep preprocessing minimal, as you asked.

One model is fitted on ALL documents jointly, so topics are shared and directly
comparable across the three sources.

Install:
    pip install bertopic sentence-transformers umap-learn hdbscan scikit-learn \
                pandas matplotlib scipy plotly

Run:
    python topic_modelling_serbian.py
"""

import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from scipy.spatial.distance import jensenshannon
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
FILES = {
    "pink": "B:\\Articles\\Vucic\\Posle_nadstresnice_ocisceno\\pink.txt",
    "rts": "B:\\Articles\\Vucic\\Posle_nadstresnice_ocisceno\\rts.txt",
    "tanjug": "B:\\Articles\\Vucic\\Posle_nadstresnice_ocisceno\\tanjug.txt",
}
OUT_DIR = Path("B:\\Articles\\Vucic\\topic_output_2")

# Multilingual embedding model (supports Serbian).
# Alternatives:
#   "BAAI/bge-m3"                      -> stronger, but larger/slower (good with a GPU)
#   "intfloat/multilingual-e5-base"    -> strong, needs "query: " prefix for best results
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

MIN_WORDS_PER_DOC = 12      # drop very short fragments
MAX_WORDS_PER_DOC = 80      # target chunk size (~150 tokens in Serbian; fits the 256 limit below)
MAX_SEQ_LENGTH = 256        # embedding model default is 128 tokens, which would truncate chunks
MIN_TOPIC_SIZE = 20         # smaller -> more, finer topics
NR_TOPICS = None            # e.g. 20 to force merging into ~20 topics; None = automatic
RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# LOADING + MINIMAL PREPROCESSING
# --------------------------------------------------------------------------- #
# Lines like "### Vesti_Najnovije_a62614-....txt" separate articles inside one file.
ARTICLE_SEPARATOR = re.compile(r"^\s*###.*$", re.M)

# Whole lines that are page boilerplate, not text. Add patterns for pink / tanjug if needed.
BOILERPLATE_PATTERNS = [
    r"^izvor\s*:.*$",            # "Izvor:RTS 1"
    r"^a post shared by.*$",     # embedded Instagram posts
    r"^https?://\S+$",
]
_BOILERPLATE = [re.compile(p, re.I) for p in BOILERPLATE_PATTERNS]

# Split after . ! ? when the next sentence starts with a capital letter
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZŠĐČĆŽ])")


def clean(text: str) -> str:
    """Very light cleaning: remove URLs and quotation marks, collapse whitespace."""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[\"“”„‟«»]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_into_docs(raw: str) -> list[str]:
    """
    Raw file -> list of documents (~MAX_WORDS_PER_DOC words each).
    - Articles are separated by '###' lines; chunks never cross article boundaries.
    - Boilerplate lines are dropped.
    - Each remaining line is a paragraph/heading; paragraphs are split into sentences,
      and sentences are greedily packed into chunks. Short headings therefore get
      attached to the text that follows them.
    """
    docs = []
    for article in ARTICLE_SEPARATOR.split(raw):
        sentences = []
        for line in article.splitlines():
            line = line.strip()
            if not line or any(rx.match(line) for rx in _BOILERPLATE):
                continue
            line = clean(line)
            if line:
                sentences += _SENT_SPLIT.split(line)

        chunk, n = [], 0
        for sent in sentences:
            w = len(sent.split())
            if chunk and n + w > MAX_WORDS_PER_DOC:
                if n >= MIN_WORDS_PER_DOC:
                    docs.append(" ".join(chunk))
                chunk, n = [], 0
            chunk.append(sent)
            n += w
        if n >= MIN_WORDS_PER_DOC:
            docs.append(" ".join(chunk))
    return docs


def load_corpus():
    docs, sources = [], []
    for name, path in FILES.items():
        raw = Path(path).read_text(encoding="utf-8", errors="ignore")
        d = list(dict.fromkeys(split_into_docs(raw)))   # drop exact duplicates within a source
        print(f"{name}: {len(d)} documents")
        docs += d
        sources += [name] * len(d)
    return docs, sources


# --------------------------------------------------------------------------- #
# SERBIAN (LATIN) STOPWORDS - used only for the topic keywords, not embeddings
# --------------------------------------------------------------------------- #
_STOP = """
i u je se da na za su sa od o to kao koji koja koje koji ali ili ne nije nisu će bi li što
iz po do kod pri prema među između ovo ova ovaj ovi ove taj ta te tu ti on ona ono oni one
mi vi ja sam si smo ste bio bila bilo bili biti ću ćeš ćemo ćete neće ima imaju imao imala
kada kad gde gdje kako jer već samo još više vrlo nakon pre posle tokom zbog kroz ovom tom
toga ovog svoj svoje svoja svog njegov njegova njegovo njen njena njeno njihov sve svi sav
sva svo nešto neki neka neko nekog ni niti pa nego dok ako iako takođe tako tamo ovde onda
sada danas juče sutra kojim kojoj kojih kome čime čemu ovim ovoj tim toj tih ovih bez protiv
oko preko iznad ispod pored uz van ka ću će može mogu moglo treba trebalo biće bude budu
rekao rekla kaže kazao izjavio izjavila istakao istakla dodao dodala naveo navela mogućnosti 
u oko iznad okolo skladno sukladno odnosno prema prema tome preko čin zapravo oglas dodan dodaj 
dodat dodao ajde hajde utjecao uticao utjecala uticala nakon poslije posle ponovo nanovo opet protiv 
ah  ali ajd aj dopustiti dopustio dopustila dopustilo sve dopusti dopušta dopuštaju dopuštam skoro 
sam uz već također takođe iako uvijek uvek sam među medju iznos vrednost jedan i najaviti još 
jedan bilo koji tko ko kako više tko što svejedno isto gdje gde odvojeno očigledno pojaviti 
se cijeniti ceniti prikladno približno zar jesu su nisu nijesu nastati okolo kao a na stranu 
pitati pitaj pitamo pitali pitao pridružen pridružena pridruženo pridruženi na nad nedostupan 
dostupni nedostupni dostupne nedostupne dostupan odsutan odsutna odsutni odsutno užasno užasan 
užasni užasne biti bio bila bilo postao postali postalo postala zbog jer postati postaje bio pre 
prije početi početak počeci počinje van iza biti vjerovati verovati ispod pored  osim najbolje 
najbolja najbolji bolje bolji bolja između izvan bi račun li s oboma obe oba  oboje dno dna br ukratko 
broj brojevi brojeva broje ali preko na a  pozivi pozivao pozivala pozvao pozvala poziv došao 
mogu ne mogu može mogu moze uzroci  uzrocnici uzročnici će ce sigurno sigurna sigurni ici ići 
idi promjene ide idu u to jasno ta ajde hajde ajd hajd dođi dolazi dolaziti dolazile dolazili 
dolazilo dodji vezi posljedično posljedicno posledično posledicno razmotriti uzimajući u obzir 
sadržati sadrzati sadržao sadržala sadrzala sadrzalo sadržalo sadrzao  sadrži sadrzi odgovara 
mogao mogla moglo mogo tečaj tecaj čas cas sat momenat trenutak trenutna trenutni trenutno dan 
dani mesec mesec godina  godine da datum cas čas časovi casovi definitivno opisati opisano opisani 
opisana opisane unatoč uprkos detalj detalji detaljan detaljni detaljna detaljne jesam nisam nismo 
drugačiji jesmo jes nije  radi radije ne radio radila gotovo nemoj dolje dole gore nagore tamo ovamo 
ovde onde tokom svaki svaka svako svakojako svakojaki svakojake efekat efekti efektno efekt efektivno 
efektivna osam osamdeset ili ej jel jedanaest drugo drugdje drugde em prazno na kraj kraju krajeve krajevi 
dovoljno potpunosti u potpuno  nepotpuno potpun nepotpun posebno poseban posebna posebni itd itd. npr čak 
ikad svaki svi svakojaki sve svugdje svugde bivši bivsi baš bas primjer primer osim ej ehej hej dalek daleko 
daleki nekoliko nekoliki nekolika petnaest peti pedeset ispuniti naći naci vatra prvi  pet popravi popravilo 
popravila popravili popravljeno prati sljedi sledi slijedi za bivši bivsi dalje četrdeset cetrdeset nađeno 
nadjeno četiri cetiri hiljade od sprijeda spreda hiljada hiljadi hiljaditi pun dalje nadalje  izdaleka njega 
ga dao dala dobiti dobili dobile dobio dati dati  davanje davali davale kreni ide idemo otišao otisao dobio 
dobila dobilo pozdrav pozdravljen pozdravljena pozdravljeno pozdravljeni  imala imao imao nije događa dogadja 
jedva ima nema nije nemoj ima utočište utociste nisam imati on ona ono će ce zdravo pomoć pomoc dakle ona ovdje 
ovde daljnjem ovdje ovde ondje onde njen ona on ono oni bok cao ćao sakrio sam njegov ovdje samo sama kući nadam 
nadati kako međutim medjutim kako kakav kakva kakvi kakvo sto takvo i takva takvi takvima trideset dvadeset  
sezdeset šezdeset sto stotina hiljadu hiljada hiljade stotine sedamdeset osamdeset devedeset ih njih nju njega 
to im ti odma odmah važnost važno vaznost vazno u utoliko doduse doduše indeks indicirati naznačeno naznaceno 
indicira informacija informacije informaciji informacijama unutarnji unutrasnji unutrašnji ukoliko umjesto 
umesto interes u izum unutra izumi pronalazak pronalasci jest je jeste nije to itd itd. njegova njegovo njegovi 
njegovima zadrži zadrzi čuva cuva zadržao zadrzao kg g km znam poznato zna ko l ml dl uglavnom zadnji zadnje 
kasnije lb t mw najmanje v manje w neka neko hajde neki vjerojatno verovatno vrvlinija malo male mali malcice 
malčice navod navodno izgleda uglavnom mnogo mozda možda ja znaci znači znaciti značiti znacio značio znacila 
značilo međuvremenu medjuvremenu mozda možda milijun milion moj gospođica gospodjica moji moje moja više štoviše 
najviše vise stovise najvise uglavnom kretanje  gospodin gospođa gospodja me ma mu mnogo moram moramo smijem smem 
smemo moje ja  na ime naime ne ni blizu skoro nužno nuzno potrebno treba trebalo trebala trebati trebao nikad ipak 
novo sljedeća sljedeca sledeća sledeca ni devet devedeset nj u niko nitko ne ipak pak niti normalno napomenuto ništa 
nista  sada nigdje nigde  dobiti dobijen očito dobiven ocito od do ugasen često cesto ugašen ugaseno ugašeno ok redu 
novo staro izostavljeno on jednom jedan jedne samo na o jedni iliti ili druga  inace oni drugo drugi inače ti trebalo 
naš nas sami izvani vani više ukupno vise dugovanje vlastito vlastita dugovanja vlastiti svoj svoji svoja stranica strana 
stranice par stranice posebna posebno pas prošlo proslo mozda moguce moguca moguci možda postavljena postavljeni postavljeno 
molimo molio molila postavljen molim plus moguće nemoguće potencijalno nemoguce pretezno pretežno prisutno vjerojatno 
prethodno verovatno najverovatnije brzo ubrzo  put prilično prilicno skoro skorije pre prije  spremno stvarno razumno 
nedavno  obzira poštovanjem povezano relativno istraživanje   članak odnosno rezultat rezultata rezultati istrazivanje 
clanak postovanjem desno levo sa rekao isto vidio recimo reci kaže kaze video odeljak drugojačije drugo drugojacije  
odjeljak videti vidjeti činiti izgledati prividno ciniti vidjeti videti sebe osjetno poslano  ozbiljno osetno sedam 
nekoliko trebao treba nece neće ona prolio prolila prolilo trebala trebao trebali pokazati pokazala pokazao pokazalo 
pokazati  str. strana značajno znacajno slično slicno pošto iskreno šest šezdeset sest sezdeset malo slicno nesto tako 
nešto  netko nekako neko neke neki nekad ponekad donekle negdje uskoro oprosti negde konkretno navedeno specificirati 
specifizirati specifican specifičan jos stop snažno snazno značajno  uspješno tako dovoljno predlažem predlazem sigurno 
uspesno sustav uspešno uzeti uzeto uzimanje rečiti reciti deset teži tezi lakši  nego hvala laksi bok to ono njihovo njihov 
njihova oni tada odatle poslije ovi oni oni debeo tanak misliti treći treci temeljito temeljno hiljadu tisucu iako tisuću tri 
trideset trista tristo kroz  tako ti napojnica tj tj. do zajedno također uzeo vrh takođe prema tr pokušao pokušala zaista pokusali 
pokušava takodje tv dvanaest dvadeset dvaput dva dve u ispod nažalost žalost zalost vjerojatno verovatno neverovatno dole gore ups 
nama koristi korišteno korisno koristeno korisno upotrebljava koristeno obično obično vrijednost razno vrednost vrlo  način mi bismo 
dobrodošli pa dobrodosli nacin otišli  bili otisli bila bilo nismo nisam što sto će ce je kada odakle jeste gde gdje poslije dok pri 
posle li koji dok hir kamo ko tko kaj cijeli celi kome god čiji ciji zašto zasto zato široko hoće voljan želi sa unutar bez nevoljan 
pobijedio čudim  neću cudim riječi svijet bi siroko hoce voljno reči reci nevoljan još jos nula nule nuli ćemo smo koje iz naše koja 
po ako nam naših kad te toga kojima pod vreme našim biće one svoje svima mora taj naša šta svih bude
srbija srbiji srbije srbijom srbijama ljudi ljudima ljude pitanje pitanja pitanjima pitan
""".split()


def _fold(s: str) -> str:
    """Strip diacritics (š->s, č->c, ć->c, ž->z, đ->dj/d) so both spellings are covered."""
    s = s.replace("đ", "dj")
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


# Words that open almost every sentence in these news texts ("Predsednik Vučić je rekao ...")
# and would otherwise dominate the topic keywords. Remove entries if you want them kept.
_DOMAIN_STOP = """
predsednik predsednika predsedniku predsednikom predsednici vučić vučića vučiću vučićem
rekao rekla rečeno izjavio izjavila istakao istakla naglasio naglasila naveo navela dodao
dodala poručio poručila zaključio zaključila ocenio ocenila kazao kazala
""".split()

STOPWORDS = sorted(set(_STOP + _DOMAIN_STOP) | {_fold(w) for w in _STOP + _DOMAIN_STOP})


# --------------------------------------------------------------------------- #
# MODEL
# --------------------------------------------------------------------------- #
def build_model() -> BERTopic:
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embedder.max_seq_length = MAX_SEQ_LENGTH

    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0,
        metric="cosine", random_state=RANDOM_STATE,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE, metric="euclidean",
        cluster_selection_method="eom", prediction_data=True,
    )
    # Topic keywords: lowercase, unigrams + bigrams, Serbian stopwords removed.
    vectorizer = CountVectorizer(
        lowercase=True, stop_words=STOPWORDS, ngram_range=(1, 2),
        min_df=3, token_pattern=r"(?u)\b[^\W\d_]{3,}\b",
    )
    return BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics=NR_TOPICS,
        top_n_words=10,
        calculate_probabilities=False,
        verbose=True,
    )


# --------------------------------------------------------------------------- #
# COMPARISON
# --------------------------------------------------------------------------- #
def compare_sources(topic_model, docs, topics, sources):
    df = pd.DataFrame({"doc": docs, "topic": topics, "source": sources})
    info = topic_model.get_topic_info().set_index("Topic")
    label = info["Name"].to_dict()

    # Share of each source's documents falling in each topic (columns sum to 1)
    counts = pd.crosstab(df["topic"], df["source"])
    share = counts / counts.sum(axis=0)
    share.index = [label[t] for t in share.index]

    # Similarity between sources' topic profiles (1 = identical, 0 = disjoint)
    names = list(share.columns)
    sim = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            sim.loc[a, b] = 1 - jensenshannon(share[a], share[b], base=2)

    return df, counts, share, sim


def plot_heatmap(share: pd.DataFrame, path: Path, top_n=25):
    # Skip the outlier topic (-1) in the plot and show the biggest topics
    plot_df = share[~share.index.str.startswith("-1")]
    plot_df = plot_df.loc[plot_df.sum(axis=1).sort_values(ascending=False).index[:top_n]]

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(plot_df) + 2))
    im = ax.imshow(plot_df.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(plot_df.shape[1]), plot_df.columns)
    ax.set_yticks(range(plot_df.shape[0]), plot_df.index, fontsize=8)
    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            ax.text(j, i, f"{plot_df.values[i, j]:.1%}", ha="center", va="center",
                    color="white", fontsize=7)
    fig.colorbar(im, ax=ax, label="Share of source's documents")
    ax.set_title("Topic distribution per source")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    docs, sources = load_corpus()
    topic_model = build_model()
    topics, _ = topic_model.fit_transform(docs)

    # Optional: reassign outliers (-1) to their nearest topic for cleaner comparison
    # topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings")
    # topic_model.update_topics(docs, topics=topics, vectorizer_model=topic_model.vectorizer_model)

    df, counts, share, sim = compare_sources(topic_model, docs, topics, sources)

    # ---- Save tables ----
    topic_model.get_topic_info().to_csv(OUT_DIR / "topics.csv", index=False)
    counts.to_csv(OUT_DIR / "topic_counts_per_source.csv")
    share.to_csv(OUT_DIR / "topic_share_per_source.csv")
    sim.to_csv(OUT_DIR / "source_similarity.csv")
    df.to_csv(OUT_DIR / "documents_with_topics.csv", index=False)

    # ---- Console summary ----
    print("\n=== Topics ===")
    print(topic_model.get_topic_info()[["Topic", "Count", "Name"]].to_string(index=False))

    print("\n=== Top 5 topics per source (share of that source's documents) ===")
    for src in share.columns:
        print(f"\n{src}")
        print(share[src].sort_values(ascending=False).head(5).map("{:.1%}".format).to_string())

    print("\n=== Topic-profile similarity between sources (1 = identical) ===")
    print(sim.round(3).to_string())

    # ---- Plots ----
    plot_heatmap(share, OUT_DIR / "topic_share_heatmap.png")

    topic_model.visualize_topics().write_html(OUT_DIR / "intertopic_map.html")
    topic_model.visualize_barchart(top_n_topics=15).write_html(OUT_DIR / "topic_words.html")
    topic_model.visualize_topics_per_class(
        topic_model.topics_per_class(docs, classes=sources)
    ).write_html(OUT_DIR / "topics_per_source.html")

    topic_model.save(str(OUT_DIR / "bertopic_model"), serialization="safetensors",
                     save_ctfidf=True, save_embedding_model=EMBEDDING_MODEL)
    print(f"\nDone. Results saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()