import os
import subprocess
import sys
import argparse

def run_command(command, quiet=False):
    try:
        if quiet:
            result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        else:
            result = subprocess.run(command, shell=True, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        if not quiet:
            print(f"Error running command: {command}")
            if e.stderr:
                print(e.stderr)
        return False

def check_remote():
    try:
        remotes = subprocess.check_output("git remote -v", shell=True).decode()
        if not remotes.strip():
            print("Error: No remote repository configured.")
            print("Please add a remote using: git remote add origin <your-repo-url>")
            return False
        return True
    except subprocess.CalledProcessError:
        print("Error: Not a git repository.")
        return False

def sync_start():
    """
    To be run when starting work: Pulls latest changes.
    """
    print("--- Starting Work: Syncing from Remote ---")
    if not check_remote():
        return

    print("\n[1/1] Pulling latest changes from remote...")
    # Try main, then master
    if not run_command("git pull origin main --rebase", quiet=True):
         print("Trying master branch...")
         run_command("git pull origin master --rebase")
    
    print("\n--- You are up to date! Happy Coding! ---")

def sync_end():
    """
    To be run when finishing work: Pushes local changes.
    """
    print("--- Finishing Work: Syncing to Remote ---")
    if not check_remote():
        return

    # 1. Pull first to avoid conflicts
    print("\n[1/4] Checking for remote updates first...")
    run_command("git pull origin main --rebase", quiet=True) or run_command("git pull origin master --rebase", quiet=True)

    # 2. Add changes
    print("\n[2/4] Staging local changes...")
    run_command("git add .")
    
    # 3. Check status
    status = subprocess.check_output("git status --porcelain", shell=True).decode()
    if not status.strip():
        print("No changes to commit.")
        return

    # 4. Commit
    commit_msg = input("\nEnter commit message: ").strip()
    if not commit_msg:
        commit_msg = "Update project"
    
    print(f"\n[3/4] Committing with message: '{commit_msg}'...")
    if run_command(f'git commit -m "{commit_msg}"'):
        # 5. Push
        print("\n[4/4] Pushing to remote...")
        if not run_command("git push origin main"):
            print("Trying master branch...")
            run_command("git push origin master")
        
    print("\n--- Changes pushed successfully! ---")

if __name__ == "__main__":
    # Ensure we are in the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    parser = argparse.ArgumentParser(description="Sync project changes.")
    parser.add_argument("action", choices=["start", "end"], help="Use 'start' when beginning work (pull) and 'end' when finishing (push).")
    
    if len(sys.argv) == 1:
        # Interactive mode if no args provided
        print("Select action:")
        print("1. Start Work (Pull changes)")
        print("2. End Work (Push changes)")
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            sync_start()
        elif choice == "2":
            sync_end()
        else:
            print("Invalid choice.")
    else:
        args = parser.parse_args()
        if args.action == "start":
            sync_start()
        elif args.action == "end":
            sync_end()
