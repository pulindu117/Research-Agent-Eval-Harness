import sys
from agent import run_agent

if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("What do you want to research? → ").strip()

    if not question:
        print("No question provided.")
        sys.exit(1)

    result = run_agent(question)

    if result.success:
        print(f"\n📄 Report saved to: {result.filename}")
    else:
        print(f"\n❌ Agent did not complete successfully. ({result.error})")
