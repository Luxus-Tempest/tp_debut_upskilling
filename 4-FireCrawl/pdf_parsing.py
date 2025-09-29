from firecrawl import Firecrawl
from dotenv import load_dotenv
import os
load_dotenv()

firecrawl = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))

response = firecrawl.scrape(
    url="https://www.pwc.com/gx/en/issues/analytics/assets/pwc-ai-analysis-sizing-the-prize-report.pdf",
    formats=["markdown"],
    parsers=["pdf"],
)

print(response.markdown)