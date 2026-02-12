import os

def generate_report():
    base_dir = "results/steganalysis"
    methods = ["cnn", "gan"]
    
    results = {}
    
    for method in methods:
        res_file = os.path.join(base_dir, method, "results.txt")
        if os.path.exists(res_file):
            with open(res_file, 'r') as f:
                content = f.read()
                # Parse accuracy
                for line in content.splitlines():
                    if "Accuracy:" in line:
                        acc = line.split(":")[1].strip()
                        results[method] = acc
        else:
            results[method] = "N/A"
            
    print("-" * 40)
    print(f"{'Appr.':<10} | {'Detection Acc.':<15}")
    print("-" * 40)
    print(f"{'CNN-Stego':<10} | {results.get('cnn', 'N/A'):<15}")
    print(f"{'GAN-Stego':<10} | {results.get('gan', 'N/A'):<15}")
    print("-" * 40)
    
    # Generate LaTeX table snippet for IEEE
    latex = f"""
\\begin{{table}}[h]
\\centering
\\caption{{Steganalysis Detection Accuracy (SRNet)}}
\\begin{{tabular}}{{|c|c|}}
\\hline
Steganography Method & Detection Accuracy \\\\
\\hline
CNN-Based & {results.get('cnn', 'N/A')} \\\\
GAN-Based & {results.get('gan', 'N/A')} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    print("\nLaTeX Table Snippet:")
    print(latex)
    
    # Save to file
    with open(os.path.join(base_dir, "final_report.txt"), "w") as f:
        f.write("Steganalysis Results\n")
        f.write(f"CNN-Stego: {results.get('cnn', 'N/A')}\n")
        f.write(f"GAN-Stego: {results.get('gan', 'N/A')}\n")
        f.write("\n")
        f.write(latex)

if __name__ == "__main__":
    generate_report()
