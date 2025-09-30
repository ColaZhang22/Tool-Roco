# ToolRoco

ToolRoco is a long-term tool-using LLM benchmark  designed for collaborative robotic tasks. It supports various task scenarios such as CabinetTask, PackGroceryTask, and SortTask.
### CabinetTask
![CabinetTask](./%20Example/CabinetTask.gif)

### PackTask
![PackTask](./%20Example/PACK.gif)

### SortTask
![SortTask](./%20Example/SortTask.gif)
---

## Project Structure

- `evaluation.py` and `evaluation_dec.py`: Scripts for evaluating task execution results.
- `prompt_template/`: Contains templates for tasks and agents.
- `real_world/`: Code for interacting with real-world robots.
- `rocobench/`: Benchmarking environment for robotic tasks.
- `Example/`: Contains example task animations and related data.
- `README.md`: Project documentation.
- Other scripts and modules: Support task execution, evaluation, and debugging.

---

## Installation

Before running the project, ensure all dependencies are installed. Use the following command:

```bash
pip install -r requirements.txt


