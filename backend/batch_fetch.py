import subprocess
import sys

categories = [
    ("cs.CL", 100),  # Computational Linguistics  
    ("cs.GR", 50),   # Computer Graphics
    ("cs.HC", 50),   # Human-Computer Interaction
    ("cs.IR", 50),   # Information Retrieval
    ("cs.CR", 50),   # Cryptography and Security
    ("cs.DB", 50),   # Databases
    ("cs.RO", 50),   # Robotics
]

def fetch_category(category, max_results):
    print(f"🔄 Fetching {max_results} papers for {category}...")
    try:
        result = subprocess.run([
            sys.executable, "fetch_papers.py", 
            "--category", category, 
            "--max_results", str(max_results)
        ], check=True, capture_output=True, text=True)
        print(f"✅ Successfully completed {category}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error fetching papers for {category}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print(f"🚀 Starting to fetch papers from {len(categories)} categories...")
    
    successful = 0
    failed = 0
    
    for category, max_results in categories:
        if fetch_category(category, max_results):
            successful += 1
        else:
            failed += 1
        print("---")
    
    print(f"🎉 Completed! {successful} successful, {failed} failed")
    
    if successful > 0:
        print("🔄 Now run: python simple_embeddings.py")
        print("🔄 Then run: python vector_database.py")

if __name__ == "__main__":
    main()
