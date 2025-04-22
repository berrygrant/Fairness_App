# reflection_and_hypotheses.py

import pandas as pd
from datetime import datetime

# ==== USER INPUT SECTION ====
student_name = "Your Name Here"
dataset_used = "compas"  # or 'stereoset'
intervention_type = "upsample"  # 'none', 'upsample', 'downsample', 'reweight'
fairness_metrics = ["Equal Opportunity", "Equal Odds"]

# ==== PROMPTS ====
def generate_reflection(student_name, dataset, intervention, fairness_metrics):
    lines = []
    lines.append(f"# Fairness in AI: Reflection Report\n")
    lines.append(f"**Student:** {student_name}")
    lines.append(f"**Dataset:** {dataset}")
    lines.append(f"**Fairness Metric(s):** {', '.join(fairness_metrics)}")
    lines.append(f"**Intervention Applied:** {intervention}\n")

    lines.append("## 1. Reflection on Hypotheses")
    lines.append("- Were your hypotheses about bias in the model supported? Why or why not?")
    lines.append("- Which groups exhibited the most unfair treatment initially?")
    lines.append("- Did the model behave as you expected?\n")

    lines.append("## 2. Effect of Intervention")
    lines.append(f"- What impact did {intervention} have on fairness metrics?")
    lines.append("- Did it reduce disparities between groups? If so, how?")
    lines.append("- What happened to accuracy, precision, and recall?\n")

    lines.append("## 3. Tradeoffs and Recommendations")
    lines.append("- Did improving fairness lower performance?")
    lines.append("- Would you deploy this model in the real world? Why or why not?")
    lines.append("- What would you do differently next time?\n")

    lines.append("---")
    lines.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    return "\n".join(lines)

# ==== GENERATE AND SAVE REPORT ====
report = generate_reflection(student_name, dataset_used, intervention_type, fairness_metrics)

with open("reflection_report.md", "w") as f:
    f.write(report)

print("Reflection report saved as 'reflection_report.md'. You can edit and submit it as needed.")
