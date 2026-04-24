# 该py用于绘制SFT过程中validation对应的BLEU和PassRate曲线

import json
import matplotlib.pyplot as plt


def main(lora_name):
    with open(f"../xllamafactory/saves/{lora_name}/trainer_state.json", "r") as f:
        data = json.load(f)

    bleu4_values = []
    passrate_values = []
    for item in data["log_history"]:
        if "eval_BLEU4" in item:
            bleu4_values.append(item["eval_BLEU4"])
        if "eval_PassRate" in item:
            passrate_values.append(item["eval_PassRate"])

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(bleu4_values, marker="o", linestyle="-")
    axs[0].set_title("Eval_BLEU4")
    axs[0].set_xlabel("Training_Steps(/100)")
    axs[0].set_ylabel("BLEU4")
    axs[1].plot(passrate_values, marker="o", linestyle="-")
    axs[1].set_title("Eval_PassRate")
    axs[1].set_xlabel("Training_Steps(/100)")
    axs[1].set_ylabel("PassRate")

    plt.tight_layout()
    plt.savefig(f"./results/{lora_name}/loss.png")
    plt.show()


if __name__ == "__main__":
    lora_name = "SFT_x88536"
    main(lora_name)