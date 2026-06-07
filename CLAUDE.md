# AskGloucester

Civic AI assistant for Gloucester, MA. Answers questions about city 
council, school committee, planning board, and other municipal documents
using RAG (Retrieval-Augmented Generation).

## Stack
- Infrastructure: Azure (Bicep IaC), deployed to rg-askgloucester-dev
- Ingestion: Python, civic-scraper, Azure Document Intelligence
- Search: Azure AI Search (hybrid keyword + vector)
- API: FastAPI on Azure Container Apps
- LLM: Azure OpenAI
- CI/CD: GitHub Actions

## Azure Resources (eastus)
- Storage Account: stakgloucesterdev (raw-documents container)
- Document Intelligence: docintel-askgloucester-dev
- AI Search: srch-askgloucester-dev
- Key Vault: kv-askgloucester-dev
- Managed Identity: id-askgloucester-dev

## Conventions
- Always use DefaultAzureCredential — never hardcode keys
- Load config from .env using python-dotenv
- Infrastructure is ephemeral — everything deployable via main.bicep
- Add clear comments explaining Azure SDK calls

## Running the pipeline
- Activate venv: source .venv/bin/activate
- Run ingestion: python run_pipeline.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD