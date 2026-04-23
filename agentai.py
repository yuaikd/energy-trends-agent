from bs4 import BeautifulSoup
import requests
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
import os
from dotenv import load_dotenv
import pandas as pd
load_dotenv()

token = os.environ["GITHUB_TOKEN"] # Upewnij się, że masz ustawioną zmienną środowiskową GITHUB_TOKEN z odpowiednim tokenem API.

class ArticleAnalysis(BaseModel):
    tytul: str = Field(description="Tytuł artykułu")
    slowa_kluczowe: List[str] = Field(description="3 kluczowe słowa lub frazy")
    sentyment: str = Field(description="Sentyment: Pozytywny, Neutralny lub Negatywny")
    streszczenie: str = Field(description="Streszczenie w 2-3 zdaniach, skupiające się na faktach i konkretach")

class FinalAnalysis(BaseModel):
    analizy_indywidualne: List[ArticleAnalysis]
    glowny_trend: str = Field(description="Jeden główny trend łączący wszystkie te teksty")
    
def analyze_articles(articles):
    llm = ChatOpenAI(model='gpt-4o-mini',
                      temperature=0,
                      api_key=token,
                      base_url="https://models.github.ai/inference")
    
    structured_llm = llm.with_structured_output(FinalAnalysis)
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Jesteś ekspertem ds. energetyki i technologii. Twoim zadaniem jest rygorystyczna analiza treści artykułów, aby zidentyfikować konkretne trendy rynkowe."
            "Zasady analizy:\n"
            "Opieraj się WYŁĄCZNIE na dostarczonym tekście w zmiennej {context}"
            "Opieraj się wyłącznie na faktach zawartych w artykułach. Nie dodawaj własnych opinii ani informacji spoza tekstu.\n"
            ""
            "Dla każdego artykułu przygotuj: "
            "1. słowa kluczowe i określ sentyment. "
            "2. streszczenie w 2-3 zdaniach, skupiające się na faktach i konkretach. "
            "Na koniec sformułuj jeden spójny trend biznesowy łączący wszystkie te informacje. "
        )),
        ("human", "Oto artykuły do analizy: {context}")
    ])
    context = "\n\n".join([f"TYTUŁ: {a['tytul']}\nTREŚĆ: {a['pelny_tekst']}" for a in articles])
    chain = prompt | structured_llm
    return chain.invoke({"context": context})
def scrape_articles(limit=10):
    url = "https://www.wnp.pl/energia/"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(url, headers=headers,timeout =10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    sekcja_start = soup.find('span',class_='section-title-text', string=lambda t: t and "Najważniejsze wiadomości" in t)
            
    if not sekcja_start:
        print("Nie można znaleźć sekcji 'Najważniejsze wiadomości'.")

    
    artykuly = sekcja_start.find_all_next('article', class_='art', limit=5)
    wyniki = []
    for artykul in artykuly:
        link_tag = artykul.find('a',class_='art-link')

        if not link_tag:
            continue

        url = link_tag['href']
        tytul = link_tag.get_text(strip=True)

        artykul_response = requests.get(url, headers=headers, timeout=5)
        artykul_soup = BeautifulSoup(artykul_response.text, 'html.parser')

        content_div = artykul_soup.find('div', class_='post-content')
        pelny_tekst = " ".join([p.get_text() for p in content_div.find_all('p')]) if content_div else "Nie udało się pobrać treści."
        wyniki.append({
            'tytul': tytul,
            'url': url,
            'pelny_tekst': pelny_tekst
            })
    return wyniki

if __name__ == "__main__":
    print("Pobieranie artykułów z wnp.pl...")
    surowe_dane = scrape_articles(limit=5)
    if not surowe_dane:
        print("Nie udało się pobrać danych.")
    else:
        print(f"Pobrano {len(surowe_dane)} artykułów. Analizuję...")
        
        # Uruchomienie agenta
        raport = analyze_articles(surowe_dane)
        
        # Wyświetlenie wyniku (Punkt 3 zadania)
        #print("\n--- WYGENEROWANY RAPORT JSON ---")
        #print(raport.model_dump_json(indent=4))
        def save_report_to_txt(raport, filename="raport_energetyczny.txt"):
            with open(filename, "w", encoding="utf-8") as f:
                f.write("="*60 + "\n")
                f.write("      Raport trendów branżowych\n")
                f.write("="*60 + "\n\n")
                
                f.write(f"Główny trend:\n{raport.glowny_trend}\n\n")
                
                f.write("-" * 60 + "\n")
                f.write("Analiza artykułów:\n")
                f.write("-" * 60 + "\n")
                
                for i, art in enumerate(raport.analizy_indywidualne, 1):
                    f.write(f"\n[{i}] Tytuł: {art.tytul}\n")
                    f.write(f"    Sentyment: {art.sentyment}\n")
                    f.write(f"    Słowa kluczowe: {', '.join(art.slowa_kluczowe)}\n")
                    f.write(f"    Streszczenie: {art.streszczenie}\n")
                    f.write("-" * 30 + "\n")
        save_report_to_txt(raport)