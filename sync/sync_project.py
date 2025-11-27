import os
import subprocess
import sys
import argparse

def run_command(command, quiet=False, capture=False):
    """
    Run a shell command with better error handling.
    """
    try:
        if capture or quiet:
            result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
            return result.stdout if capture else True
        else:
            result = subprocess.run(command, shell=True, check=True, text=True)
            return True
    except subprocess.CalledProcessError as e:
        if not quiet:
            print(f"❌ Error running command: {command}")
            if e.stderr:
                print(e.stderr)
        return False

def get_current_branch():
    """
    Get the current git branch name.
    """
    try:
        branch = subprocess.check_output("git branch --show-current", shell=True, text=True).strip()
        return branch if branch else "main"
    except subprocess.CalledProcessError:
        return "main"

def check_remote():
    """
    Check if git remote is configured.
    """
    try:
        remotes = subprocess.check_output("git remote -v", shell=True, text=True).strip()
        if not remotes:
            print("❌ Error: No remote repository configured.")
            print("Please add a remote using: git remote add origin <your-repo-url>")
            return False
        print(f"✓ Remote configured")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error: Not a git repository.")
        return False

def sync_start():
    """
    To be run when starting work: Pulls latest changes.
    """
    print("=" * 50)
    print("🚀 Starting Work: Syncing from Remote")
    print("=" * 50)
    
    if not check_remote():
        return False

    branch = get_current_branch()
    print(f"\n📍 Current branch: {branch}")
    print(f"\n[1/2] Fetching latest changes from remote...")
    
    if not run_command("git fetch origin", quiet=True):
        print("❌ Failed to fetch from remote")
        return False
    
    print(f"[2/2] Pulling changes into {branch}...")
    
    # Try to pull with rebase
    if run_command(f"git pull origin {branch} --rebase", quiet=True):
        print(f"✅ Successfully pulled latest changes from {branch}")
    else:
        print(f"⚠️  No changes to pull or branch doesn't exist on remote yet")
    
    print("\n" + "=" * 50)
    print("✅ You are up to date! Happy Coding!")
    print("=" * 50)
    return True

def sync_end():
    """
    To be run when finishing work: Pushes local changes.
    """
    print("=" * 50)
    print("📤 Finishing Work: Syncing to Remote")
    print("=" * 50)
    
    if not check_remote():
        return False

    branch = get_current_branch()
    print(f"\n📍 Current branch: {branch}")

    # 1. Pull first to avoid conflicts
    print(f"\n[1/5] Checking for remote updates...")
    run_command("git fetch origin", quiet=True)
    
    # Check if there are remote changes
    try:
        local = subprocess.check_output(f"git rev-parse {branch}", shell=True, text=True).strip()
        remote = subprocess.check_output(f"git rev-parse origin/{branch}", shell=True, text=True, stderr=subprocess.DEVNULL).strip()
        
        if local != remote:
            print("⚠️  Remote has changes, pulling first...")
            if not run_command(f"git pull origin {branch} --rebase", quiet=True):
                print("❌ Failed to pull remote changes. Please resolve conflicts manually.")
                return False
    except subprocess.CalledProcessError:
        print("ℹ️  Branch doesn't exist on remote yet (first push)")

    # 2. Add changes
    print(f"\n[2/5] Staging local changes...")
    run_command("git add .")
    
    # 3. Check status
    try:
        status = subprocess.check_output("git status --porcelain", shell=True, text=True).strip()
        if not status:
            print("ℹ️  No changes to commit.")
            print("\n" + "=" * 50)
            print("✅ Already up to date!")
            print("=" * 50)
            return True
        
        # Show what will be committed
        print("\n📝 Files to be committed:")
        for line in status.split('\n'):
            print(f"   {line}")
            
    except subprocess.CalledProcessError:
        print("❌ Failed to check git status")
        return False

    # 4. Commit
    print(f"\n[3/5] Creating commit...")
    commit_msg = input("\n💬 Enter commit message (or press Enter for default): ").strip()
    if not commit_msg:
        commit_msg = "Update project"
    
    if not run_command(f'git commit -m "{commit_msg}"'):
        print("❌ Failed to create commit")
        return False
    
    print(f"✅ Committed: {commit_msg}")

    # 5. Push
    print(f"\n[4/5] Pushing to remote ({branch})...")
    
    # Try to push
    push_cmd = f"git push origin {branch}"
    if not run_command(push_cmd, quiet=True):
        # If push fails, try with -u flag (for first push)
        print("Trying with upstream flag...")
        push_cmd = f"git push -u origin {branch}"
        if not run_command(push_cmd):
            print("❌ Failed to push changes")
            return False
    
    print(f"✅ Successfully pushed to origin/{branch}")
    
    print(f"\n[5/5] Verifying sync...")
    print("✅ All changes synced successfully!")
    
    print("\n" + "=" * 50)
    print("✅ Changes pushed successfully!")
    print("=" * 50)
    return True

if __name__ == "__main__":
    # Ensure we are in the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    print(f"📂 Working directory: {project_root}\n")

    parser = argparse.ArgumentParser(
        description="Sync project changes with Git.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sync/sync_project.py start    # Pull latest changes
  python sync/sync_project.py end      # Push your changes
  python sync/sync_project.py          # Interactive mode
        """
    )
    parser.add_argument(
        "action", 
        nargs='?',
        choices=["start", "end"], 
        help="Use 'start' when beginning work (pull) and 'end' when finishing (push)."
    )
    
    args = parser.parse_args()
    
    if not args.action:
        # Interactive mode if no args provided
        print("Select action:")
        print("  1. Start Work (Pull changes)")
        print("  2. End Work (Push changes)")
        choice = input("\nEnter 1 or 2: ").strip()
        if choice == "1":
            sync_start()
        elif choice == "2":
            sync_end()
        else:
            print("❌ Invalid choice.")
            sys.exit(1)
    else:
        if args.action == "start":
            success = sync_start()
        elif args.action == "end":
            success = sync_end()
        
        sys.exit(0 if success else 1)
