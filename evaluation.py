import os
import argparse
import re
from natsort import natsorted
import json

# from MassGen.massgen.backend import response

def evaluateCurrentStep(all_json, metrics, tools_info):

    # calculate valid metrics
    for index in range(len(all_json)):
        metrics["total_steps"] += 1
        if all_json[index]["Execution"] == 1:
            metrics["thought_valid"] += 1
            metrics["tool_calling_valid"] += 1
            metrics["parameter_valid"] += 1
            metrics["execution_valid"] += 1
        elif all_json[index]["Valid Parameter"] == 1:
            metrics["thought_valid"] += 1
            metrics["tool_calling_valid"] += 1
            metrics["parameter_valid"] += 1
        elif all_json[index]["Tools Calling"] == 1:
            metrics["thought_valid"] += 1
            metrics["tool_calling_valid"] += 1

        # calcluate all tools information
        if all_json[index]["Tools Calling"] == 1:
            tool_match = re.search(r"<tool calling>(.*?)</tool calling>", all_json[index]["Response"], re.DOTALL)
            assert tool_match, "No <tool calling> section found in response"
            tool_text = tool_match.group(1).strip()
            tool_dict = json.loads(tool_text)
            for agent in tool_dict:
                if tool_dict[agent] == {}:
                    tools_info["WAIT"] += 1
                    continue
                tool_name = tool_dict[agent]["tool"]
                if tool_name in tools_info.keys():
                    tools_info[tool_name] += 1
                else:
                    tools_info[tool_name] = 1
                if tool_name == "CONNECT_AGENT" or tool_name == "DISCONNECT_AGENT":
                    metrics["organization"] += 1

        # calculate modification and reflection
        reflection_flag = False
        mofification_flag = False
        if index > 0:
            if all_json[index]["Tools Calling"] != all_json[index-1]["Tools Calling"]:
                reflection_flag = True
            if all_json[index]["Valid Parameter"] != all_json[index-1]["Valid Parameter"]:
                reflection_flag = True
            if all_json[index]["Execution"] != all_json[index-1]["Execution"]:
                reflection_flag = True
            if all_json[index]["Reason"] != all_json[index-1]["Reason"]:
                reflection_flag = True
            if reflection_flag:
                metrics["reflection"] += 1

            if all_json[index]["Tools Calling"] > all_json[index-1]["Tools Calling"]:
                mofification_flag = True
            if all_json[index]["Valid Parameter"] > all_json[index-1]["Valid Parameter"]:
                mofification_flag = True
            # if all_json[index]["Execution"] > all_json[index-1]["Execution"]:
            if "Execution" in all_json[index].keys() and "Execution" not in all_json[index-1].keys():
                mofification_flag = True
            if mofification_flag:
                metrics["modification"] += 1

    # calculate prob
    metrics["modification_prob"] = metrics["modification"] / metrics["total_steps"]
    metrics["reflection_prob"] = metrics["reflection"] / metrics["total_steps"]
    metrics["tool_calling_valid_prob"] = metrics["tool_calling_valid"] / metrics["total_steps"]
    metrics["parameter_valid_prob"] = metrics["parameter_valid"] / metrics["total_steps"]
    metrics["execution_valid_prob"] = metrics["execution_valid"] / metrics["total_steps"]
    
def evaluation(args):
    folder_path = args.folder_path
    metrics = {
        "task_success": 0,
        "thought_valid": 0,
        "tool_calling_valid": 0,
        "parameter_valid": 0,
        "execution_valid": 0,
        "modification": 0,
        "reflection": 0,
        "total_steps": 0,
        "organization": 0,
    }
    tools_info = {
        "CONNECT_AGENT": 0,
        "DISCONNECT_AGENT": 0,
        "MOVE": 0,
        "PICK": 0,
        "PLACE": 0,
        "OPEN": 0,
        "CLOSE": 0,
        "WAIT": 0,

    }
    # Read all episodes
    run_folders = natsorted([
        run_name for run_name in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, run_name)) and run_name.startswith("run_")
    ])
    for run_name in run_folders:
        all_json = []
        run_path = os.path.join(folder_path, run_name)

        # Read all turns in one episode
        step_folders = natsorted([
            step_name for step_name in os.listdir(run_path)
            if os.path.isdir(os.path.join(run_path, step_name)) and step_name.startswith("step_")
        ])
        for step_name in step_folders:
            step_path = os.path.join(run_path, step_name, "log")
            json_files = natsorted([
                file_name for file_name in os.listdir(step_path)
                if file_name.endswith(".json")
            ])
            for file_name in json_files:
                log_path = os.path.join(step_path, file_name)
                with open(log_path, "r") as f:
                    log_content = json.load(f)
                    all_json.append(log_content)

            # Start Evaluate Current Step
            evaluateCurrentStep(all_json, metrics, tools_info)

    # Print Evaluation Results
    print("Evaluation Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print("Tools Information:")
    for key, value in tools_info.items():
        print(f"  {key}: {value}")
    
    folder_name = os.path.basename(os.path.normpath(folder_path))
    with open(f"{folder_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{folder_name}_tools_info.json", "w") as f:
        json.dump(tools_info, f, indent=2)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, default="/Users/zhangke/Desktop/robot-collab-toolusing/aamas/Tool-roco/Experiment/aamas_pack_gpt_4o_mini_cen_org_re")
    args = parser.parse_args()
    evaluation(args)
   
