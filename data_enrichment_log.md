# Data Enrichment Log

Record every manual data addition here. Use the template below for each entry.

---
**Date:** 2026-01-28  
**Collected By:** GitHub Copilot  
**Record Type:** observation  
**Indicator / Event Name:** Account Ownership Rate (2022)  
**Source Name:** World Bank Global Findex / WDI  
**Source URL:** https://api.worldbank.org/v2/country/ETH/indicator/FX.OWN.TOTL.ZS?format=json  
**Original Text / Figure:** "Ethiopia - Account ownership at a financial institution or with a mobile-money-service provider (% of population ages 15+)" shows 46.48 for 2022.  
**Confidence:** medium  
**Notes & Rationale:** Adds an in-between observation to capture the post-2021 yet pre-2024 account ownership trajectory for trend modeling.

---
**Date:** 2026-01-28  
**Collected By:** GitHub Copilot  
**Record Type:** observation  
**Indicator / Event Name:** Mobile Subscription Penetration (2022)  
**Source Name:** World Bank WDI  
**Source URL:** https://api.worldbank.org/v2/country/ETH/indicator/IT.CEL.SETS.P2?format=json  
**Original Text / Figure:** "Mobile cellular subscriptions (per 100 people)" equals 56.9643 for Ethiopia in 2022.  
**Confidence:** medium  
**Notes & Rationale:** Serves as an access-enabler proxy to relate device availability with account and usage indicators.

---
**Date:** 2026-01-28  
**Collected By:** GitHub Copilot  
**Record Type:** observation  
**Indicator / Event Name:** Individuals Using the Internet (2021)  
**Source Name:** World Bank WDI  
**Source URL:** https://api.worldbank.org/v2/country/ETH/indicator/IT.NET.USER.ZS?format=json  
**Original Text / Figure:** "Individuals using the Internet (% of population)" records 16.698 for 2021.  
**Confidence:** medium  
**Notes & Rationale:** Internet usage is a leading indicator for digital payment readiness and will feed into usage regressions.

---
**Date:** 2026-01-28  
**Collected By:** GitHub Copilot  
**Record Type:** event  
**Indicator / Event Name:** AfDB Digital Payments Initiative Support (2023-10-31)  
**Source Name:** African Development Bank  
**Source URL:** https://www.afdb.org/en/news-and-events/press-releases/afdb-supports-ethiopia-digital-payments-initiative-2023-10-31  
**Original Text / Figure:** Press release announces AfDB support to modernize Ethiopia's instant retail payments and agent networks.  
**Confidence:** medium  
**Notes & Rationale:** Captures donor-financed infrastructure push that should influence usage and crossover metrics after go-live.

---
**Date:** 2026-01-28  
**Collected By:** GitHub Copilot  
**Record Type:** impact_link  
**Indicator / Event Name:** AfDB initiative effect on P2P/ATM crossover  
**Source Name:** African Development Bank  
**Source URL:** https://www.afdb.org/en/news-and-events/press-releases/afdb-supports-ethiopia-digital-payments-initiative-2023-10-31  
**Original Text / Figure:** Funding package is aimed at scaling instant retail payments, QR acceptance, and merchant onboarding.  
**Confidence:** medium  
**Notes & Rationale:** Links the infrastructure investment to an expected increase in USG_CROSSOVER with a 12-month lag for implementation.

---
**Date:** YYYY-MM-DD  
**Collected By:** Your Name  
**Record Type:** observation | event | impact_link | target  
**Indicator / Event Name:**  
**Source Name:**  
**Source URL:**  
**Original Text / Figure:**  
**Confidence:** high | medium | low  
**Notes & Rationale:** Why this addition matters for the forecasting tasks.

---
