import json

def save_results_to_json(new_results, alg):
        with open(f'results/{alg}.json', 'r') as json_file:
                json_content = json.load(json_file)
        json_content.append(new_results)
        with open(f'results/{alg}.json', 'w') as json_file:
                json.dump(json_content, json_file)