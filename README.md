# Vucic.Topic.Modeling-
Vucic.Topic.Modeling

Text classification, BERTopic, visuals.

A small research pipeline for comparing how different Serbian news outlets cover the same period. It scrapes articles from vucic.rs, prepares the text, and fits a single BERTopic model across three source corpora (Pink, RTS, Tanjug) so that topics are shared and directly comparable between outlets.

Pipeline overview
Scrapper.py  ──►  Vucic_text_analysis.py  ──►  cyrilic2alphabet.py  ──►  topic_modelling.py
 scrape            merge / count tokens /       Cyrillic → Latin          BERTopic + comparison
 articles          split by "Извор" (source)    transcription             of sources + visuals
Step	Script	What it does
1	Scrapper.py	Scrapes articles from the Najnovije (latest news) archive on vucic.rs.
2	Vucic_text_analysis.py	Merges the scraped .txt files, counts tokens, and splits articles by whether they carry a source line (Извор:).
3	cyrilic2alphabet.py	Provides transcribe(), which converts Serbian Cyrillic to Latin script.
4	topic_modelling.py	Chunks the texts, fits BERTopic, compares topic profiles across outlets, and writes tables and interactive plots.
Scripts in detail
Scrapper.py — article scraper
Opens https://vucic.rs/n12/Vesti/Najnovije with Playwright (headless Chromium).
Repeatedly clicks the visible «Учитајте више» ("Load more") button (a.js-btn-load-more) and collects article URLs matching /Vesti/<section>/a<id>-<slug>.html.
Downloads each new article with requests, extracts the title (<h1>) and paragraph text with BeautifulSoup, and stops at "related news" headings.
Saves one .txt file per article, formatted as:
  URL: <article url>
  TITLE: <title>

  <body text>
Skips articles that already exist on disk, so it can be re-run safely.
Stops when the button disappears, after MAX_STAGNANT_CLICKS clicks in a row produce nothing new, or when MAX_ARTICLES is reached.
Writes listing_checkpoint.html (a snapshot of the listing page) to the output folder.

Key settings (top of the file):

Setting	Default	Meaning
OUTPUT_DIR	B:\Articles\Vucic	Where article files are saved
MAX_ARTICLES	5000	Upper limit (None for no limit)
CRAWL_DELAY	1.5 s	Pause between article requests
LOAD_MORE_WAIT	3 s	Wait after each "Load more" click
MAX_STAGNANT_CLICKS	3	Stop after this many clicks with no new articles
Vucic_text_analysis.py — merging and source statistics

For two folders (Posle_nadstresnice and Pre_nadstresnice, i.e. articles from after and before the Novi Sad railway station canopy collapse):

Merges all .txt files in each folder into one file (merged1.txt, merged2.txt).
Counts tokens in each merged file using tiktoken (cl100k_base).
Splits articles into with_izvor.txt and without_izvor.txt depending on whether they contain an Извор: ("Source:") line, and prints how often each source (e.g. RTS, Tanjug, Pink) is cited.
cyrilic2alphabet.py — Cyrillic → Latin

transcribe(text) maps every Serbian Cyrillic letter to its Latin equivalent (including digraphs lj, nj, dž) and preserves capitalisation. Characters outside the Cyrillic alphabet are left unchanged.

python
from cyrilic2alphabet import transcribe

transcribe("ће сутра поднети потписе")
# 'će sutra podneti potpise'
topic_modelling.py — BERTopic and source comparison

Fits one BERTopic model on all documents jointly, then compares how each outlet's coverage is distributed over the shared topics.

Why BERTopic? Serbian is highly inflected, and classic bag-of-words topic models (LDA/NMF) need good lemmatisation, for which there is no lightweight, reliable tool for Serbian Latin. Multilingual transformer embeddings capture semantic similarity despite different surface forms, so preprocessing can stay minimal.

Processing steps

Load three files: pink.txt, rts.txt, tanjug.txt (Serbian, Latin script).
Clean & chunk
Articles are separated by ### lines; chunks never cross article boundaries.
Boilerplate lines (e.g. Izvor:..., embedded Instagram posts, bare URLs) and quotation marks are removed.
Text is split into sentences and packed into chunks of about 80 words (minimum 12); exact duplicates within a source are dropped.
Embed with paraphrase-multilingual-mpnet-base-v2 (max_seq_length = 256).
Cluster with UMAP (5 components, cosine) and HDBSCAN (min_cluster_size = 20).
Label topics with c-TF-IDF over unigrams and bigrams, using a large Serbian stopword list (Latin script, with and without diacritics) plus domain stopwords (e.g. "predsednik", "Vučić", "rekao").
Compare sources
Share of each source's documents falling in each topic.
Pairwise similarity of topic profiles: 1 − Jensen–Shannon distance (1 = identical, 0 = disjoint).

Outputs (written to OUT_DIR)

File	Content
topics.csv	Topic overview (id, size, name, keywords)
topic_counts_per_source.csv	Document counts per topic and source
topic_share_per_source.csv	Share of each source's documents per topic
source_similarity.csv	Jensen–Shannon-based similarity between sources
documents_with_topics.csv	Every document with its assigned topic and source
topic_share_heatmap.png	Heatmap of topic distribution per source
intertopic_map.html	Interactive intertopic distance map
topic_words.html	Bar charts of top words for the largest topics
topics_per_source.html	Topic frequencies by source
bertopic_model/	Saved BERTopic model (safetensors)

Main settings: EMBEDDING_MODEL, MIN_WORDS_PER_DOC, MAX_WORDS_PER_DOC, MIN_TOPIC_SIZE (smaller → more, finer topics), NR_TOPICS (None = automatic), RANDOM_STATE. Stronger multilingual alternatives such as BAAI/bge-m3 are noted in the code.

Installation

Python 3.10+ is recommended.

bash
# Scraping
pip install playwright requests beautifulsoup4
playwright install chromium

# Text analysis
pip install tiktoken

# Topic modelling
pip install bertopic sentence-transformers umap-learn hdbscan scikit-learn \
            pandas matplotlib scipy plotly

A GPU is not required but speeds up embedding considerably, especially with larger models.

Usage

Note: all paths are currently hard-coded Windows paths (B:\Articles\Vucic\...). Edit them at the top of each script before running.

bash
# 1. Scrape articles
python Scrapper.py

# 2. Merge files, count tokens, split by source line
python Vucic_text_analysis.py

# 3. (Used as a helper) Cyrillic → Latin transcription
#    from cyrilic2alphabet import transcribe

# 4. Fit the topic model and compare sources
python topic_modelling.py

The scraped data and the per-outlet files (pink.txt, rts.txt, tanjug.txt) are not included in this repository. Preparing them (sorting articles by outlet, transcribing to Latin script, and light cleaning) happens between steps 2 and 4.

Repository structure
.
├── README.md
├── Scrapper.py               # Playwright + BeautifulSoup scraper
├── Vucic_text_analysis.py    # merging, token counts, source statistics
├── cyrilic2alphabet.py       # Cyrillic → Latin transcription
└── topic_modelling.py        # BERTopic model, source comparison, visuals
Responsible use

The scraper is rate-limited by default (CRAWL_DELAY). Before running it, check the target site's terms of service and robots.txt, keep the delay generous, and do not redistribute scraped article text without permission. Topic models describe patterns in text; they do not by themselves establish bias or intent, so interpret the results with care.

License

N/O
