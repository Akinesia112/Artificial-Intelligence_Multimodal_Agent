from __future__ import annotations
from autogen import ConversableAgent, register_function
import os, sys, re, ast, math
from typing import Dict, List, get_type_hints

SCORE_KEYWORDS: dict[int, list[str]] = {
    1: ["awful", "horrible", "disgusting"],
    2: ["bad", "unpleasant", "offensive"],
    3: ["average", "uninspiring", "forgettable"],
    4: ["good", "enjoyable", "satisfying"],
    5: ["awesome", "incredible", "amazing"]
}

# ────────────────────────────────────────────────────────────────
# 0. OpenAI API key setup ── *Do **not** modify this block.*
# ────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    sys.exit("❗ Set the OPENAI_API_KEY environment variable first.")
LLM_CFG = {"config_list": [{"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}]}

# ────────────────────────────────────────────────────────────────
# 1. Utility data structures & helper functions
# ────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()

def fetch_restaurant_data(restaurant_name: str) -> dict[str, list[str]]:
    data = {}
    target = normalize(restaurant_name)
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            name, review = line.split('.', 1)
            if normalize(name) == target:
                data.setdefault(name.strip(), []).append(review.strip())
    return data


def calculate_overall_score(restaurant_name: str, food_scores: List[int], customer_service_scores: List[int]) -> dict[str, str]:
    """Geometric-mean rating rounded to 3 dp."""
    n = len(food_scores)
    if n == 0 or n != len(customer_service_scores):
        raise ValueError("food_scores and customer_service_scores must be non-empty and same length")
    total = sum(((f**2 * s)**0.5) * (1 / (n * (125**0.5))) * 10 for f, s in zip(food_scores, customer_service_scores))
    return {restaurant_name: f"{total:.3f}"}

def parse_restaurant_name(query: str) -> str:
    """Extract the restaurant’s name from queries like
       “How good is the restaurant Taco Bell overall?”"""
    # 1) “restaurant <Name> overall”
    m = re.search(r"restaurant\s+(.+?)\s+overall", query, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 2) “overall score for <Name>” (handles hyphens, with or without “?” or “.”)
    m = re.search(r"overall\s+score\s+for\s+(.+?)(?:[\?\.]|$)", query, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 3) general fallback “for <Name>” up to punctuation or end of string
    m = re.search(r"for\s+(?:the\s+)?(.+?)(?:[\?\.]|$)", query, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return query.strip()

def extract_scores_positional(reviews: List[str]) -> tuple[List[int], List[int]]:
    """For each review, pick the first two matched keywords by position
       (food first, service second), defaulting to the same score if only one."""
    def keyword_positions(text: str):
        positions = []
        text_low = text.lower()
        for score, kws in SCORE_KEYWORDS.items():
            for kw in kws:
                for m in re.finditer(rf"\b{re.escape(kw)}\b", text_low):
                    positions.append((m.start(), score))
        return sorted(positions, key=lambda x: x[0])

    food_scores, service_scores = [], []
    for rev in reviews:
        pos = keyword_positions(rev)
        if len(pos) >= 2:
            f, s = pos[0][1], pos[1][1]
        elif len(pos) == 1:
            f = s = pos[0][1]
        else:
            continue
        food_scores.append(f)
        service_scores.append(s)
    return food_scores, service_scores


# register functions
fetch_restaurant_data.__annotations__ = get_type_hints(fetch_restaurant_data)
calculate_overall_score.__annotations__ = get_type_hints(calculate_overall_score)

# ──────────────────────────────────────────────
# 2. Agent setup
# ──────────────────────────────────────────────

def build_agent(name, msg):
    return ConversableAgent(name=name, system_message=msg, llm_config=LLM_CFG)

DATA_FETCH = build_agent(
    "fetch_agent",
    "Input is {'Name': [...reviews...]}. Each review has 2 adjectives (food, service).\n"
    "Given a user query like “How good is Taco Bell?”, "
    "reply with exactly:\n"
    '{"call":"fetch_restaurant_data","args":{"restaurant_name":"<restaurant>"}}'
)
ANALYZER = build_agent(
    "review_analyzer_agent",
    "Input: {'<Restaurant Name>': [<list of review strings>]}.  \n"
    "Each review contains two adjectives (one describing food, one service).  \n"
    "Use the SCORE_KEYWORDS map to turn each into a pair of ints.  \n"
    "Reply *only* with:\n\n"
        "food_scores=[...]\n"
        "customer_service_scores=[...]"
        f"{SCORE_KEYWORDS}"
)
SCORER = build_agent(
    "scoring_agent",
        "Input: restaurant name + two lists of ints.  \n"
        "Reply *only* with exactly:\n"
        "calculate_overall_score(\"<restaurant>\", food_scores, customer_service_scores)"
)
ENTRY = build_agent(
    "entry",
    "You are the orchestrator. Coordinate the other agents to produce the final result."
    )

# register functions
register_function(
    fetch_restaurant_data,
    caller=DATA_FETCH,
    executor=ENTRY,
    name="fetch_restaurant_data",
    description="Fetch reviews from specified data file by name.",
)
register_function(
    calculate_overall_score,
    caller=SCORER,
    executor=ENTRY,
    name="calculate_overall_score",
    description="Compute final rating via geometric mean.",
)


# ────────────────────────────────────────────────────────────────
# 3. Conversation helpers
# ────────────────────────────────────────────────────────────────

def run_chat_sequence(entry: ConversableAgent, sequence: list[dict]) -> str:
    ctx = {**getattr(entry, "_initiate_chats_ctx", {})}
    for step in sequence:
        msg = step["message"].format(**ctx)
        chat = entry.initiate_chat(
            step["recipient"], message=msg,
            summary_method=step.get("summary_method", "last_msg"),
            max_turns=step.get("max_turns", 2),
        )
        out = chat.summary
        # Data fetch output
        if step["recipient"] is DATA_FETCH:
            for past in reversed(chat.chat_history):
                try:
                    data = ast.literal_eval(past["content"])
                    if isinstance(data, dict) and data and not ("call" in data):
                        ctx.update({"reviews_dict": data, "restaurant_name": next(iter(data))})
                        break
                except:
                    continue
        # Analyzer output passed directly
        elif step["recipient"] is ANALYZER:
            ctx["analyzer_output"] = out
    return out

def orchestrate(query: str) -> dict[str, str]:
    ctx = {"user_query": query}

    # 1) fetch_agent → fetch_restaurant_data
    fetch_resp = ENTRY.initiate_chat(
        DATA_FETCH,
        message="Find reviews for this: {user_query}".format(**ctx),
        max_turns=2
    )
    # extract the dict of reviews returned by the function
    ctx["reviews_dict"] = fetch_resp.tool_response  # AutoGen unwraps it for you
    ctx["restaurant_name"] = next(iter(ctx["reviews_dict"]))

    # 2) review_analyzer_agent → parse into two score lists
    reviews = ctx["reviews_dict"][ctx["restaurant_name"]]
    analyzer_input = {"Name": reviews}
    analyze_resp = ENTRY.initiate_chat(
        ANALYZER,
        message=str({ctx["restaurant_name"]: reviews}),
        max_turns=2
    )
    # AutoGen will capture the two lists from the assistant response:
    ctx.update(analyze_resp.summary_vars)  # expect keys: 'food_scores', 'customer_service_scores'

    # 3) scoring_agent → calculate_overall_score
    score_resp = ENTRY.initiate_chat(
        SCORER,
        message=(
            f"{ctx['restaurant_name']} | "
            f"food_scores={ctx['food_scores']} | "
            f"customer_service_scores={ctx['customer_service_scores']}"
        ),
        max_turns=2
    )
    return score_resp.tool_response  # this is your final {name: "X.XXX"}

ConversableAgent.initiate_chats = lambda self, seq: run_chat_sequence(self, seq)

# ──────────────────────────────────────────────
# 4. Main entry
# ──────────────────────────────────────────────

def main(user_query: str, data_path: str = "restaurant-data.txt"):
    global DATA_PATH
    DATA_PATH = data_path
    
    # 1. Figure out which restaurant they asked about
    name_input = parse_restaurant_name(user_query)
    
    # 2. Read all of its reviews
    data = fetch_restaurant_data(name_input)
    if not data:
        raise ValueError(f"No reviews found for '{name_input}'")
    # use the actual key as it appears in the file
    restaurant_key = next(iter(data))
    reviews = data[restaurant_key]
    
    # 3. Turn each review into (food_score, service_score)
    food_scores, service_scores = extract_scores_positional(reviews)

    # 4. Compute the final geometric-mean score
    result = calculate_overall_score(restaurant_key, food_scores, service_scores)

    # Print & return so test.py can capture both
    print(result)
    return result
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python main.py path/to/data.txt "How good is Subway?" ')
        sys.exit(1)

    path = sys.argv[1]
    query = sys.argv[2]
    main(query, path)

