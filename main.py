from dotenv import load_dotenv
load_dotenv()
import os

def main():
    print("Hello from langchain-course!")
    print(os.getenv("OPENAI_API_KEY"))
    print(os.getenv("GOOGLE_API_KEY"))


if __name__ == "__main__":
    main()
