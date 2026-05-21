import subprocess
import sys

# Mapping of package names to their common import names if different
PKG_MAP = {
    "langchain-community": "langchain_community",
    "langchain-huggingface": "langchain_huggingface",
    "langchain-ollama": "langchain_ollama",
    "langchain-openai": "langchain_openai",
    "python-dotenv": "dotenv",
    "sentence-transformers": "sentence_transformers"
}

def is_installed(package):
    base_name = package.split('>=')[0].split('==')[0].split('>')[0].split('<')[0].split('~=')[0].strip()
    import_name = PKG_MAP.get(base_name, base_name.replace('-', '_'))
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        try:
            # Fallback: check with pip show
            subprocess.check_call([sys.executable, "-m", "pip", "show", base_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except (subprocess.CalledProcessError, AttributeError):
            return False

with open('requirements.txt', 'r') as f:
    deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]

for dep in deps:
    if is_installed(dep):
        print(f"Skipping {dep}, already installed.")
        continue
        
    print(f"Installing {dep}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "--quiet"])
        print(f"Successfully installed {dep}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {dep}: {e}")
