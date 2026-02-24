import os

def load_system_prompt(name: str = None):
    """Load a named system prompt from file."""
    system_prompt_file = os.path.join(os.path.dirname(__file__), "..", "prompts", "system_prompts.txt")
    with open(system_prompt_file, 'r') as f:
        content = f.read()
    prompts = {}
    current_name = None
    current_lines = []
    for line in content.split('\n'):
        if line.startswith('===PROMPT:') and line.endswith('==='):
            if current_name is not None:
                prompts[current_name] = '\n'.join(current_lines).strip()
            current_name = line[10:-3].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_name is not None:
        prompts[current_name] = '\n'.join(current_lines).strip()
    if name not in prompts:
        available = list(prompts.keys())
        raise ValueError(f"System prompt '{name}' not found. Available: {available}")
    return prompts[name]