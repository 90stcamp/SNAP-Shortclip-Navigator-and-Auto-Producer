import openai
import json
from tqdm import tqdm
from datasets import load_dataset


with open('key.json', 'r') as f:
    openai.api_key = json.load(f)['openai_key']

system="You are helpful AI to do tasks."


completion = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system}
    ],
    temperature=0,
    max_tokens=256,
)


def refine(title: str, temperature=0, max_tokens=256):
    prompt = f"""
    script: {title}

    Summarize the above script focusing on the main plot.
    """

    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return completion.choices[0].message.content


if __name__ == '__main__':
    dataset = load_dataset('big_patent', codes=["d"], version="1.0.0")
    
    for sen in tqdm(dataset['train'][0]):
        output=refine(sen)
        print(output)

