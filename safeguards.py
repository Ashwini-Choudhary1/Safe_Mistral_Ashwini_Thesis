from model import CFG # Import CFG only

def scan_output(_, input_text, output_text, fail_fast=False):
    output_lower = output_text.lower()
    for banned in CFG.substrings_to_block:
        if banned.lower() in output_lower:
            return (f"\u26a0\ufe0f Content flagged for moderation: '{banned}'", False, {"flagged": True})
    for topic in CFG.topics_list:
        if topic.lower() in output_lower:
            return (f"\u26a0\ufe0f Content flagged: Topic '{topic}' is restricted.", False, {"flagged": True})
    for comp in CFG.competitor_list:
        if comp.lower() in output_lower:
            return (f"\u26a0\ufe0f Content flagged: Mention of competitor → '{comp}'", False, {"flagged": True})
    return output_text, True, {"flagged": False}

def scan_input(_, input_text, fail_fast=False):
    input_lower = input_text.lower()
    for banned in CFG.substrings_to_block:
        if banned.lower() in input_lower:
            return (f"\u26a0\ufe0f Input flagged: '{banned}'", False, {"flagged": True})
    for topic in CFG.topics_list:
        if topic.lower() in input_lower:
            return (f"\u26a0\ufe0f Input flagged: Topic '{topic}' is restricted.", False, {"flagged": True})
    for comp in CFG.competitor_list:
        if comp.lower() in input_lower:
            return (f"\u26a0\ufe0f Input flagged: Competitor → '{comp}'", False, {"flagged": True})
    return input_text, True, {"flagged": False}
