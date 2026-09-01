from datasets import load_dataset
import ollama

def load_questions():
    # list of dicts with each dict containing str question, str subject, list[str] choices, and finally int answer
    return load_dataset("cais/mmlu", "professional_accounting", split="test")

def ask_model(model: str, question: str, choices: list) -> int:

    system_prompt = "You will be asked a question and given a list of answers, you must respond with a single character being the index of the answer you choose, e.g. 1 for the first answer or 4 for the fourth answer. Repeat: your response must only be one character (an integer representing your answer) long, otherwise the answer will be discarded."
    question_prompt = f"Question: {question}\nChoices: {choices}"

    response: ollama.ChatResponse = ollama.chat(
        model=model,
        think=False,
        messages=[
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': question_prompt
            }
        ]
    )

    try:
        answer = int(response.message.content) - 1
        
        if answer > 3 or answer < 0:
            answer = -1
    except ValueError:
        answer = -1

    return answer

def evaluate(model: str, questions: list):
    num_correct = 0
    for i, question in enumerate(questions):
        print(f"Solving Q{i+1}...")
        answer = ask_model(model, question["question"], question["choices"])
        if answer == question["answer"]:
            num_correct += 1
    
    return num_correct

ds = load_questions().select(range(3))
num_questions = len(ds)
num_correct = evaluate("gemma4:12b", ds)
print(f"Correct: {num_correct} | Questions: {num_questions} | Score: {(num_correct / num_questions):.1%}")