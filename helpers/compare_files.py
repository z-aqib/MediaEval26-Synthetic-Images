# This script compares two Python files side by side
# It creates an HTML file where differences are highlighted in color

from pathlib import Path
import difflib
import webbrowser

# Change these paths to the two files you want to compare
file2_path = Path("notebooks/03_clip_vit_experiments.py")
file1_path = Path("notebooks/01_efficientnet_b0_experiments.py")

# Output HTML file where the side-by-side comparison will be saved
output_path = Path("helpers/file_comparison_diff.html")

# Make sure the helpers folder exists
output_path.parent.mkdir(parents=True, exist_ok=True)

# Read both files as lists of lines
file1_lines = file1_path.read_text(encoding="utf-8").splitlines()
file2_lines = file2_path.read_text(encoding="utf-8").splitlines()

# Create a side-by-side HTML diff
html_diff = difflib.HtmlDiff(
    tabsize=4,
    wrapcolumn=120
).make_file(
    fromlines=file1_lines,
    tolines=file2_lines,
    fromdesc=str(file1_path),
    todesc=str(file2_path),
    context=True,      # Shows only changed sections with some surrounding lines
    numlines=5         # Number of unchanged lines shown around each difference
)

# Save the HTML comparison file
output_path.write_text(html_diff, encoding="utf-8")

# Open the HTML file automatically in your browser
webbrowser.open(output_path.resolve().as_uri())

print(f"Side-by-side comparison saved to: {output_path}")