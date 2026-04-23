# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    # Create an LLM.
    llm = LLM(model="")
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params, cook_flag=[0, 1])
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)
    outputs = llm.generate(["Who is Bob?", "Who is Alice?"], sampling_params, cook_flag=[0, 1])
    
    outputs = llm.generate(["Who is Baaff?", "faf afa fasf?"], sampling_params, cook_flag=[0, 0])
    
    outputs = llm.generate(["Who is rsr5?", "Who is Peggs2ter?", "AB is Peggs2ter?", "Who ifafa Peggs2ter?"], sampling_params, cook_flag=[1, 1, 1, 1])


if __name__ == "__main__":
    main()
