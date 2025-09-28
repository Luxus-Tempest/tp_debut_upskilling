import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from pypdf import PdfReader
from utils import print_header, print_separator, get_user_input

load_dotenv()
# Read the PDF file and extract text from all pages
pdf = PdfReader("1-agno-pdf/pwc-ai-analysis.pdf")
full_text = "".join(page.extract_text() or "" for page in pdf.pages)

def always_return_full_pdf(agent, query, num_documents=None, ** kwargs):
    return [{"content": full_text, "meta_data": {"source": "pwc-ai-analysis.pdf"}}]

# Initialize the agent with the xAI model and the retriever function
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    knowledge=None,
    search_knowledge=True,
    knowledge_retriever=always_return_full_pdf,
)

print("Agent ready! ")

def interactive_chat():
    """Run interactive chat"""
    print_header (" AI PDF Assistant")
    print("Ask questions about your PDF document")
    print("Type 'quit' to exit, 'help' for commands")

    try:
        while True:
            question = get_user_input(" Your question: ")
            if question. lower () == "quit":
                print(" Goodbye!")
                break
            elif question. lower () == "help":
                print("\nCommands:")
                print("- Ask any question about the PDF")
                print("- 'quit' to exit")
                continue
            elif not question:
                continue

            print ("An Thinking")
            print_separator ()
            agent.print_response(question, stream=True)
            print_separator ()

    except Exception as e:
        print(f"X Error: {e}")

if __name__ == "__main__":
    interactive_chat()