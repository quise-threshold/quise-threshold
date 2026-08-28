def install_package(branch_name):
    import subprocess
    import sys
    # Run the pip install command
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", 
             f"git+https://dfeldmeyer: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@github.com/dfeldmeyer/global_sensitivity.git@{branch_name}"],
            check=True
        )
        print("Package successfully installed.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install the package: {e}")