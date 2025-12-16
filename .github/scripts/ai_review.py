import os
import subprocess
import requests
import json
import sys

# --- CONFIGURATION ---
OPEN_WEBUI_URL = os.environ.get("OPEN_WEBUI_URL") # e.g., https://chat.myserver.com
JWT_TOKEN = os.environ.get("OPEN_WEBUI_TOKEN")
MODEL = "qwen3-coder:30b"

def get_pr_diff():
    """Calculates the diff between the PR branch and the base branch."""
    base_ref = os.environ.get("GITHUB_BASE_REF", "main")
    
    try:
        # Fetch the base branch to ensure we can diff against it
        subprocess.run(["git", "fetch", "origin", base_ref], check=True, stderr=subprocess.DEVNULL)
        
        # Get the diff (excluding lock files to save tokens)
        cmd = [
            "git", "diff", 
            f"origin/{base_ref}...HEAD", 
            "--", ":!package-lock.json", ":!yarn.lock", ":!*.svg"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error getting diff: {e}")
        return None

def get_ai_review(diff):
    if not diff:
        return "No changes detected."
    
    if len(diff) > 30000:
        return "⚠️ The diff is too large for the AI to review comfortably. Please break down the PR."

    # Open WebUI (OpenAI Compatible) Endpoint
    api_endpoint = f"{OPEN_WEBUI_URL}/api/chat/completions"

    headers = {
        "Authorization": f"Bearer {JWT_TOKEN}",
        "Content-Type": "application/json"
    }

    prompt = (
        "You are a strict Senior Code Reviewer. Review the following git diff.\n"
        "1. Prioritize LOGIC BUGS, SECURITY FLAWS, and PERFORMANCE ISSUES.\n"
        "2. Do not nitpick simple formatting/whitespace unless it breaks the code.\n"
        "3. Be concise and actionable.\n"
        "4. If the code is good, simply respond with 'LGTM'.\n\n"
        f"```diff\n{diff}\n```"
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful code review assistant."},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    try:
        response = requests.post(api_endpoint, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"❌ AI Connection Error: {str(e)}"

if __name__ == "__main__":
    if not OPEN_WEBUI_URL or not JWT_TOKEN:
        print("Error: Missing secrets OPEN_WEBUI_URL or OPEN_WEBUI_TOKEN")
        sys.exit(1)

    diff = get_pr_diff()
    review = get_ai_review(diff)
    print(review)
