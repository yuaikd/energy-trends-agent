# Agent AI - Analiza trendów branżowych
Automatyczne narzędzie do scrapowania i inteligentnej analizy trendów w sektorze energetycznym. Projekt stworzony z myślą o wsparciu analitycznym w branży paliwowo-energetycznej (Case Study: Orlen).

## O projekcie
Agent automatycznie pobiera najnowsze wiadomości z portalu (wnp.pl), a następnie wykorzystuje model **GPT-4o-mini** do:
* Ekstrakcji kluczowych faktów i słów kluczowych.
* Analizy sentymentu wypowiedzi.
* Syntetyzowania rozproszonych informacji w jeden spójny trend biznesowy.

## Technologie
* **Język:** Python 3.14.2
* **LLM Orchestration:** LangChain
* **Web Scraping:** BeautifulSoup4
* **Modele:** GPT-4o-mini (GitHub Models API)
* **Walidacja danych:** Pydantic (Structured Output)

## 📋 Jak to działa?
1. **Scraper** pobiera 5 najnowszych artykułów z sekcji "Energia".
2. **Parser** czyści tekst i przygotowuje kontekst dla modelu.
3. **LLM** przetwarza dane zgodnie ze zdefiniowanym schematem `ArticleAnalysis`.
4. **Generator** tworzy raport końcowy w formacie `.txt` (opcjonalnie JSON).