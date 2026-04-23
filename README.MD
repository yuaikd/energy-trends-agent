# Agent AI - Analiza trendów branżowych
Automatyczne narzędzie do scrapowania i inteligentnej analizy trendów w sektorze energetycznym. Projekt stworzony z myślą o wsparciu analitycznym w branży paliwowo-energetycznej (Case Study: Orlen).
## Wygenerowany raport przez agenta 23.04.2026
<details>
  <summary>Kliknij, aby zobaczyć raport </summary>

  ```text
============================================================
      Raport trendów branżowych
============================================================

Główny trend:
W Polsce następuje intensyfikacja inwestycji w energetykę jądrową oraz odnawialne źródła energii (OZE), co jest wspierane przez programy rządowe i nowe regulacje. Wzrost znaczenia technologii jądrowych oraz integracja OZE w systemy zarządzania energią wskazują na dążenie do zrównoważonego rozwoju i stabilności energetycznej kraju.

------------------------------------------------------------
Analiza artykułów:
------------------------------------------------------------

[1] Tytuł: Ta polska spółka dostanie pieniądze na przygotowanie do udziału w budowie elektrowni atomowej
    Sentyment: Pozytywny
    Słowa kluczowe: finansowanie, elektrownia jądrowa, Mostostal Kielce
    Streszczenie: Mostostal Kielce otrzyma 13,7 mln zł na przygotowanie do budowy pierwszej elektrowni jądrowej w Polsce w ramach programu ARP 'Atom bez barier'. Inwestycja ma na celu wzmocnienie krajowego przemysłu oraz dostosowanie zakładu do standardów sektora jądrowego, co pozwoli firmie na skuteczną konkurencję na rynkach międzynarodowych.
------------------------------

[2] Tytuł: Jak przystąpić do wirtualnej elektrowni OZE?
    Sentyment: Pozytywny
    Słowa kluczowe: Wirtualna Elektrownia, OZE, UNIMOT
    Streszczenie: Wirtualna Elektrownia UNIMOT integruje instalacje OZE, umożliwiając ich zarządzanie jak jedną dużą elektrownią. System optymalizuje sprzedaż energii, co pozwala właścicielom instalacji na stabilne przychody i minimalizację ryzyka finansowego, a także zwiększa efektywność kontraktacji na rynku energii.
------------------------------

[3] Tytuł: Europa przyznaje się do epokowego błędu. Polska przed decyzją
    Sentyment: Neutralny
    Słowa kluczowe: energetyka jądrowa, Polska, decyzje administracyjne
    Streszczenie: Ursula von der Leyen przyznała, że Europa popełniła błąd, odchodząc od energetyki jądrowej, co w Polsce prowadzi do budowy pierwszej elektrowni jądrowej w Choczewie. Wybór technologii dla drugiej elektrowni jest obecnie kwestią polityczno-administracyjną, a eksperci podkreślają potrzebę dywersyfikacji technologii jądrowych.
------------------------------

[4] Tytuł: Jest projekt noweli obejmujący pracowników prywatnych kopalń system osłon
    Sentyment: Neutralny
    Słowa kluczowe: nowelizacja, górnictwo, wsparcie dla pracowników
    Streszczenie: Projekt nowelizacji ustawy o górnictwie węgla kamiennego ma na celu zapewnienie równego dostępu do świadczeń osłonowych dla pracowników prywatnych kopalń. Nowe przepisy umożliwią im korzystanie z urlopów górniczych i odpraw pieniężnych po przeniesieniu do spółek państwowych, co ma na celu łagodzenie skutków likwidacji kopalń.
------------------------------

[5] Tytuł: "Paląca potrzeba" w energetyce. Chodzi o systemy wsparcia
    Sentyment: Neutralny
    Słowa kluczowe: systemy wsparcia, OZE, dekarbonizacja
    Streszczenie: Renata Mroczek z URE podkreśliła potrzebę analizy systemów wsparcia dla OZE, aby ocenić, które technologie powinny być wspierane. Wskazała na konieczność stabilności regulacyjnej oraz wsparcia publicznego dla projektów dekarbonizacyjnych, które są kluczowe dla osiągnięcia celów klimatycznych.
------------------------------

```

</details>

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

## Jak to działa?
1. **Scraper** pobiera 5 najnowszych artykułów z sekcji "Energia".
2. **Parser** czyści tekst i przygotowuje kontekst dla modelu.
3. **LLM** przetwarza dane zgodnie ze zdefiniowanym schematem `ArticleAnalysis`.
4. **Generator** tworzy raport końcowy w formacie `.txt` (opcjonalnie JSON).
