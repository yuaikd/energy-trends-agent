# AI Agent - Industry Trend Analysis
An automated tool for web scraping and intelligent trend analysis within the energy sector. Designed to provide analytical support for the fuel and power industry (Case Study: Orlen).
## Agent Report (April 23, 2026)
<details>
  <summary>Click to expand the report</summary>

```text
============================================================
       Industry Trends Report
============================================================

Main trend:
Poland is experiencing an intensification of investments in nuclear energy and renewable energy sources (RES), supported by government programs and new regulations. The growing importance of nuclear technologies and the integration of RES into energy management systems indicate a drive toward sustainable development and national energy stability.

------------------------------------------------------------
Article Analysis:
------------------------------------------------------------

[1] Title: This Polish company will receive funds to prepare for participation in the construction of a nuclear power plant
    Sentiment: Positive
    Keywords: financing, nuclear power plant, Mostostal Kielce
    Summary: Mostostal Kielce will receive PLN 13.7 million to prepare for the construction of Poland's first nuclear power plant under the ARP 'Atom without barriers' program. The investment aims to strengthen the national industry and adapt the plant to nuclear sector standards, allowing the company to compete effectively on international markets.
------------------------------

[2] Title: How to join a virtual RES power plant?
    Sentiment: Positive
    Keywords: Virtual Power Plant, RES, UNIMOT
    Summary: The UNIMOT Virtual Power Plant integrates RES installations, enabling them to be managed as one large power plant. The system optimizes energy sales, allowing installation owners to achieve stable revenues and minimize financial risk, while also increasing contracting efficiency on the energy market.
------------------------------

[3] Title: Europe admits to an epochal error. Poland faces a decision
    Sentiment: Neutral
    Keywords: nuclear energy, Poland, administrative decisions
    Summary: Ursula von der Leyen admitted that Europe made a mistake by moving away from nuclear energy, which in Poland is leading to the construction of the first nuclear power plant in Choczewo. The choice of technology for the second power plant is currently a political and administrative issue, and experts emphasize the need for diversification of nuclear technologies.
------------------------------

[4] Title: There is a draft amendment covering private mine workers with a social safety net
    Sentiment: Neutral
    Keywords: amendment, mining, employee support
    Summary: The draft amendment to the hard coal mining act aims to ensure equal access to protective benefits for private mine workers. New regulations will enable them to use mining leaves and severance payments after transferring to state-owned companies, which aims to mitigate the effects of mine closures.
------------------------------

[5] Title: "Burning need" in energy. It’s about support systems
    Sentiment: Neutral
    Keywords: support systems, RES, decarbonization
    Summary: Renata Mroczek from the Energy Regulatory Office (URE) emphasized the need to analyze support systems for RES to assess which technologies should be supported. She pointed to the need for regulatory stability and public support for decarbonization projects, which are key to achieving climate goals.
------------------------------

```

</details>

## About the Project
The agent automatically fetches the latest news from the wnp.pl portal and utilizes the GPT-4o-mini model to:

* Extract key facts and technical keywords.
* Analyze the sentiment of industry-specific statements.
* Synthesize fragmented data into a single, coherent business trend.

## Tech Stack
* **Language:** Python 3.14.2

* **LLM Orchestration:** LangChain

* **Web Scraping:** BeautifulSoup4

* **Models:** GPT-4o-mini (GitHub Models API)

* **Data Validation:** Pydantic (Structured Output)

## How It Works
1. **Scraper:** Collects the 5 most recent articles from the "Energy" section.

2. **Parser:** Cleans the text and prepares the context for the model.

3. **LLM:** Processes data according to the defined `ArticleAnalysis` schema.

4. **Generator:** Produces the final report in `.txt` format (JSON optional).
