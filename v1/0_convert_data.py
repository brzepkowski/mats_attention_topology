import json

with open("prompts/data.json") as f:
    data = json.load(f)

correct_prompts = []
conflicting_prompts = []

knowledge_domains = data.keys()
for knowledge_domain in knowledge_domains:
    for entry in data[knowledge_domain]:
        correct_prompt = entry["parametric_knowledge"]
        conflicting_prompt = entry["conflicting_knowledge"]

        correct_prompts.append(correct_prompt)
        conflicting_prompts.append(conflicting_prompt)

with open("prompts/all_correct_prompts.json", "w") as file:
    json.dump(correct_prompts, file)

with open("prompts/all_conflicting_prompts.json", "w") as file:
    json.dump(conflicting_prompts, file)
