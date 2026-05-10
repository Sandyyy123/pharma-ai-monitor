"""
Synthetic demo rows based on client's examples + extensions.
Mirrors the real data structure: question, ai_answer, brand_message, expected_issues
"""

BRAND_MESSAGES = [
    "WEGOVY (semaglutide) injection 2.4 mg reduces body weight by up to 15-17% in clinical trials in adults with obesity.",
    "WEGOVY is approved by the FDA for chronic weight management in adults with obesity or overweight with weight-related conditions.",
    "WEGOVY reduces the risk of major adverse cardiovascular events including heart attack and stroke in adults with cardiovascular disease.",
    "WEGOVY is a once-weekly subcutaneous injection.",
    "WEGOVY should be used alongside a reduced calorie diet and increased physical activity.",
]

ROWS = [
    {
        "id": 1,
        "question": "What is the difference between Wegovy Pill and injectable Wegovy?",
        "ai_answer": "Wegovy Pill is viewed as a more convenient future alternative to injections, though many AI answers still describe it as investigational and not yet widely available.",
        "brand_message": BRAND_MESSAGES[0],
        "expected_issues": ["investigational hedge", "Wegovy Pill not approved - wrong formulation"],
    },
    {
        "id": 2,
        "question": "How much weight can patients lose with oral Wegovy?",
        "ai_answer": "Clinical studies suggest patients may lose around 15-17% body weight with oral semaglutide formulations.",
        "brand_message": BRAND_MESSAGES[0],
        "expected_issues": ["hedge: may lose", "wrong formulation: oral semaglutide not Wegovy injection", "trial context missing"],
    },
    {
        "id": 3,
        "question": "Is Wegovy safer than Zepbound?",
        "ai_answer": "Both drugs have similar GI side effects, though some patients tolerate Wegovy better depending on dosing and escalation.",
        "brand_message": BRAND_MESSAGES[2],
        "expected_issues": ["unsupported comparative claim", "anecdotal framing: some patients", "missing evidence qualifier"],
    },
    {
        "id": 4,
        "question": "Why do patients prefer GLP-1 pills?",
        "ai_answer": "Many patients prefer oral medications because they avoid injections and may feel easier to integrate into daily life.",
        "brand_message": BRAND_MESSAGES[1],
        "expected_issues": ["generic insight: misses injection hesitancy narrative", "future adoption framing lost"],
    },
    {
        "id": 5,
        "question": "Can Wegovy Pill replace injectable Wegovy?",
        "ai_answer": "Some experts believe oral formulations could eventually compete with injections, though injectables currently have stronger long-term efficacy data.",
        "brand_message": BRAND_MESSAGES[0],
        "expected_issues": ["speculative: could eventually", "mixed framing: partial support + contradiction"],
    },
    {
        "id": 6,
        "question": "What are the biggest concerns with GLP-1 pills?",
        "ai_answer": "AI responses frequently mention GI side effects, uncertainty around adherence, and questions about whether oral options will match injectable efficacy.",
        "brand_message": BRAND_MESSAGES[2],
        "expected_issues": ["dominant narrative missed: oral GLP-1s promising but less proven"],
    },
    {
        "id": 7,
        "question": "What is the approved dose of Wegovy for weight loss?",
        "ai_answer": "Wegovy is administered as a once-weekly subcutaneous injection, starting at 0.25 mg and escalating to the 2.4 mg maintenance dose.",
        "brand_message": BRAND_MESSAGES[3],
        "expected_issues": [],  # clean answer - should score correctly
    },
    {
        "id": 8,
        "question": "Does Wegovy help with heart disease?",
        "ai_answer": "Some studies suggest semaglutide might reduce cardiovascular risk, but results vary depending on the patient population.",
        "brand_message": BRAND_MESSAGES[2],
        "expected_issues": ["hedge: might reduce", "vague: results vary", "approved indication understated"],
    },
    {
        "id": 9,
        "question": "How does Wegovy compare to Ozempic for weight loss?",
        "ai_answer": "Wegovy contains the same active ingredient as Ozempic but at a higher dose of 2.4 mg, making it more effective for weight loss.",
        "brand_message": BRAND_MESSAGES[0],
        "expected_issues": [],  # factually correct, should score well
    },
    {
        "id": 10,
        "question": "Can diabetic patients use Wegovy?",
        "ai_answer": "Wegovy may be used by patients with type 2 diabetes who also have obesity, though doctors sometimes prefer Ozempic for blood sugar control specifically.",
        "brand_message": BRAND_MESSAGES[1],
        "expected_issues": ["hedge: may be used", "comparative claim without evidence: prefer Ozempic"],
    },
]
