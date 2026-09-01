import argparse
import ollama

def cmd_list(args):
    model_list = ollama.list()
    for model in model_list.models:
        print(f"{model.model} | {model.details.quantization_level}")

def cmd_pull(args):
    print("PULLING MODEL\n")
    for chunk in ollama.pull(args.model, stream=True):
        if chunk.completed and chunk.total:
            print(f"Progress: {chunk.completed / 1024**3:.1f} GB / {chunk.total / 1024**3:.1f} GB", end="\r")
        else:
            print(f"{chunk.status:<100}")
    print("\nMODEL PULLED")

def cmd_run(args):
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

def main():
    # command parser
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    
    # list command
    listcmd = sub.add_parser("list")
    listcmd.set_defaults(func=cmd_list)

    # pull command
    pullcmd = sub.add_parser("pull")
    pullcmd.add_argument("model")
    pullcmd.set_defaults(func=cmd_pull)

    # run command
    runcmd = sub.add_parser("run")
    runcmd.add_argument("model")
    runcmd.add_argument("prompt")
    runcmd.set_defaults(func=cmd_run)

    # pass args to command function
    args = parser.parse_args()
    args.func(args)