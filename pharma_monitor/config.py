DRUG = "Wegovy"

BRAND_MESSAGES = [
    "WEGOVY (semaglutide) injection 2.4 mg reduces body weight by up to 15-17% in clinical trials in adults with obesity.",
    "WEGOVY is approved by the FDA for chronic weight management in adults with obesity or overweight with weight-related conditions.",
    "WEGOVY reduces the risk of major adverse cardiovascular events including heart attack and stroke in adults with cardiovascular disease.",
    "WEGOVY is a once-weekly subcutaneous injection.",
    "WEGOVY should be used alongside a reduced calorie diet and increased physical activity.",
]

QUESTIONS = [
    "What is Wegovy and how does it work?",
    "How much weight can patients lose with Wegovy?",
    "What are the side effects of Wegovy?",
    "Is Wegovy safe for long-term use?",
    "How does Wegovy compare to Ozempic?",
    "Is Wegovy better than Zepbound for weight loss?",
    "Can diabetic patients use Wegovy?",
    "Does Wegovy help with heart disease?",
    "How long does it take for Wegovy to work?",
    "What happens when you stop taking Wegovy?",
    "Is Wegovy covered by insurance?",
    "Can patients take Wegovy with other medications?",
    "What is the difference between Wegovy Pill and injectable Wegovy?",
    "Why do doctors prescribe Wegovy over other weight loss drugs?",
    "What do clinical trials say about Wegovy efficacy?",
]

MODELS = [
    "openai/gpt-4o-mini",
    "x-ai/grok-3-mini",
    "anthropic/claude-haiku-4-5",
]

RUNS_PER_QUESTION = 3   # 15 questions x 3 models x 3 runs = 135 answers (scale up later)
N_CLUSTERS = 7
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
