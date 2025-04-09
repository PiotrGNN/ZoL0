
#!/usr/bin/env python3
"""
Script to install pip packages locally in a Replit environment
"""
import os
import subprocess
import sys
from pathlib import Path

# Directory to install packages to
LOCAL_LIBS_DIR = "python_libs"

# Packages that need to be installed locally (not available in nixpkgs)
PACKAGES = [
    "pybit==2.4.1",
]

def create_libs_dir():
    """Create the local libs directory if it doesn't exist"""
    Path(LOCAL_LIBS_DIR).mkdir(exist_ok=True)
    # Create an empty __init__.py to make the directory a package
    Path(f"{LOCAL_LIBS_DIR}/__init__.py").touch(exist_ok=True)
    
def install_package(package):
    """Install a package to the local directory"""
    print(f"Installing {package} to {LOCAL_LIBS_DIR}...")
    result = subprocess.run(
        [
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "--user",
            package
        ],
        check=False,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error installing {package}:")
        print(result.stderr)
        return False
    
    print(f"Successfully installed {package}")
    return True

def main():
    """Main function to set up the environment"""
    print("Setting up local packages environment...")
    create_libs_dir()
    
    # Create .gitignore for python_libs if it doesn't exist
    gitignore_path = Path(".gitignore")
    if gitignore_path.exists():
        with open(gitignore_path, "r") as f:
            content = f.read()
        if LOCAL_LIBS_DIR not in content:
            with open(gitignore_path, "a") as f:
                f.write(f"\n# Local packages directory\n{LOCAL_LIBS_DIR}/\n")
    else:
        with open(gitignore_path, "w") as f:
            f.write(f"# Local packages directory\n{LOCAL_LIBS_DIR}/\n")
    
    # Install each package
    for package in PACKAGES:
        install_package(package)
    
    print("\nLocal packages setup complete!")
    print(f"To use these packages, add this to the beginning of your scripts:")
    print("import sys")
    print(f"sys.path.insert(0, \"{LOCAL_LIBS_DIR}\")")

if __name__ == "__main__":
    main()
