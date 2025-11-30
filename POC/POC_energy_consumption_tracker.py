from codecarbon import EmissionsTracker

def process_prompt(prompt: str):
    return ""

def main():
    prompt = "your prompt here"

    tracker = EmissionsTracker()
    tracker.start()

    # Code to measure
    output = process_prompt(prompt)

    emissions = tracker.stop()

    print("Output:", output)
    print(f"Energy consumed: {emissions} kWh")

if __name__ == "__main__":
    main()
