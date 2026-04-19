from datasets import Dataset

data = {
    "question": [
        "What was Infosys revenue in 2024?",
        "Who is the CEO of Infosys?",
        "What are Infosys business segments?"
    ],
    "ground_truth": [
        "Infosys revenue in 2024 was ₹153,670 crore.",
        "The CEO of Infosys is Salil Parekh.",
        "Infosys operates in segments like Financial Services, Retail, Manufacturing."
    ]
}

dataset = Dataset.from_dict(data)