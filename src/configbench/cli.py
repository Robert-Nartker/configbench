import argparse
import ollama

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    
    sub.add_parser("list")
    pullcmd = sub.add_parser("pull")
    pullcmd.add_argument("model")
    runcmd = sub.add_parser("run")
    runcmd.add_argument("model")
    runcmd.add_argument("prompt")

    args = parser.parse_args()
    
    if args.command == "list":
        model_list = ollama.list()
        for model in model_list.models:
            print(f"{model.model} | {model.details.quantization_level}")
    elif args.command == "pull":
        print("PULLING MODEL\n")
        for chunk in ollama.pull(args.model, stream=True):
            if chunk.completed and chunk.total:
                print(f"Progress: {chunk.completed / 1024**3:.1f} GB / {chunk.total / 1024**3:.1f} GB", end="\r")
            else:
                print(f"{chunk.status:<100}")
        print("\nMODEL PULLED")
    elif args.command == "run":
        print("RUNNING MODEL\n")
        
        stream = ollama.chat(
            model=args.model,
            messages=[
                {'role': 'user', 'content': args.prompt}
            ],
            stream=True
        )

        print(">> ", end='')

        for chunk in stream:
            print(chunk.message.content, end='', flush=True)

        print("\n")