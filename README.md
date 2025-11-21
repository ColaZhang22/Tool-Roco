# ToolRoCo

**ToolRoCo** is a multi-turn, tool-using LLM benchmark designed for collaborative robotic tasks, based on **RoCo**  
<https://github.com/MandiZhao/robot-collab>.

ToolRoCo treats all agents as tools and aims to explore the **self-organization ability** of LLM agents. Currently, ToolRoCo supports both **open-source** and **closed-source** models, and includes three collaborative tasks:

- **Cabinet**  
- **PackGrocery**  
- **Sort**

The following is a demo obtained from experiments conducted with **GPT-5**.
### <center>CabinetTask</center>
![CabinetTask](./%20Example/CabinetTask.gif)

### <center>PackTask</center>
![PackTask](./%20Example/PACK.gif)

### <center>SortTask</center>
![SortTask](./%20Example/SortTask.gif)
---

## Cooperative Paradigm
ToolRoCo provides four cooperation paradigms:
| Cooperation Paradigm | Centralized LLM           | Decentralized LLMs       |
|:------------------:|:------------------------:|:-----------------------:|
| **Agent-not-as-Tool** | Centralized             | Decentralized           |
| **Agent-as-Tool**     | Centralized Self-organization | Self-organization       |

---
## Tools and Prompt of ToolRoCo
The tool list for each agent can be found at `Tool-Roco/prompt_template/agent/Agent_name/tools.json`. You can add different tools to the tool list for each agent as needed.  

The prompt templates for tasks are located in `Tool-Roco/prompt_template/task/TaskName`.  

Here is an example of a cooperative tool:

```json
{
  "type": "function",
  "function": {
    "name": "CONNECT_AGENT",
    "description": "When you cannot finish this task by yourself alone, use this function to add other agent into agent pool to help you. You can share your current objective or needs with one or multiple agents, providing only new and relevant information without repeating what others have already said.",
    "parameters": {
      "type": "object",
      "properties": {
        "agent_name": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "A list of agent names you want to tell. You may include one or multiple agents."
        }
      },
      "required": ["agent_name"]
      }
    }
  }
  ```
## Installation

Before running the project, ensure all dependencies are installed. Use the following command:

```bash
pip install -r requirements.txt
```

## Run
For **open-source models**, use `os_centralized.py` and `os_decentralized.py`.  
For **closed-source models**, use `run_centralized.py` and `run_decentralized.py`.  

To start the agent as a tool mechanism, modify the `COMM_MODE` parameter in the bash script:  
- For centralized setups, change `COMM_MODE` from `"centralized"` to `"auto_organization"` in `os_centralized.py` and `run_centralized.py`.  
- For decentralized setups, change `COMM_MODE` from `"decentralized"` to `"auto_organization"` in `os_decentralized.py` and `run_decentralized.py`.

Here is an example:
```bash
bash ./os_central.sh
```

