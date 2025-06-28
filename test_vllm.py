import argparse
import json
import requests

def read_file(file_path: str) -> str:
    """Read the content of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        raise SystemExit(f"Error: File '{file_path}' not found.")
    except Exception as e:
        raise SystemExit(f"Error reading file: {e}")

def send_request(content: str, model: str, api_url: str, max_tokens: int, 
                 top_k: int, top_p: float, temperature: float, 
                 repetition_penalty: float) -> None:
    """Send a request to the vLLM API and print the response."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "stream": False
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        print(json.dumps(result, indent=2))
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"Request failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Send file content to vLLM API")
    parser.add_argument("--file", required=True, help="Path to the input file")
    parser.add_argument("--model", default="facebook/opt-1.3b", help="Model name")
    parser.add_argument("--host", default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K sampling")
    parser.add_argument("--top-p", type=float, default=0.6, help="Top-P sampling")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.05, help="Repetition penalty")

    args = parser.parse_args()
    api_url = f"http://{args.host}:{args.port}/v1/chat/completions"

    content = read_file(args.file)
    send_request(
        content=content,
        model=args.model,
        api_url=api_url,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty
    )

if __name__ == "__main__":
    main()  