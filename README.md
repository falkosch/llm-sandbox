# llm-sandbox

A Python sandbox project to explore and experiment with text generation using Large Language Models (LLMs).

## Repository Structure

The repository is structured as follows:

```
pytorch-project/
├── data/                   # Directory to store chat JSON files
├── src/
│   ├── __init__.py         # Shared code
│   ├── *.py                # Entrypoints for model demonstration
├── requirements*.txt       # Python dependencies
├── setup-venv.sh           # Setup script for the .venv
```

## Prerequisites

- **Python Version**: Ensure Python version **3.10.x** or **3.11.x** is installed. (Python versions >= 3.13 are currently not compatible with PyTorch.)
- A performant GPU with Cuda-support is desirable, but you can fall back to using CPU if you want.

## Installation

1. Clone this repository:

   ```bash
   $ git clone <repository-url>
   ```

2. Set up a virtual environment and install dependencies:

   ```bash
   $ bash setup-venv.sh
   ```

3. Create a `.env` file with Hugging Face credentials.

## Hugging Face Credentials

This project uses the [transformers](https://huggingface.co/docs/transformers/index) python package for downloading and using pre-trained models. To ensure smooth functionality:

1. Obtain an [access token](https://huggingface.co/settings/tokens) from your Hugging Face account.
2. Create a `.env` file in the root directory with the following content:

    ```dotenv
    HF_TOKEN=<your access token>
    HF_HOME=.cache/
    ```

   Ensure that:

    - You have requested gated model access where required (e.g., for models like LLaMA).

## How to Use

Once all dependencies are set up, you can run any of the demo scripts located in the `src` directory:

```bash
$ python src/<script-name>.py
```

### Example:

```bash
$ python src/recurrentgemma.py
```

> **Note**: The file `__init__.py` contains shared code and is not meant to be executed directly.

## Data Handling

The `data/` directory contains `.json5` files, which store structured chat-like message interactions. The following schema is used for these files:

```
[
    {
        "role": "system" | "user" | "assistant",
        "content": "message text"
    },
    ...
]
```

### Message Schema Details:

1. **Roles**:
    - **`system`**: Defines the model's behavior or initial instructions. Not every model supports `system` roles (e.g., `src/gemma-*.py` and `src/recurrentgemma.py` do not use these entries).
    - **`user`**: Represents user-supplied input for the model.
    - **`assistant`**: Represents the model-generated output in response to the user.

2. **Content**: The `content` field contains the text for each message, whether it's input or output.

### Usage Guidelines:

- A chat session may optionally begin with a **`system`** role.
- The first non-`system` message **must** have the role **`user`**.
- Messages should alternate between the **`user`** and **`assistant`** roles.
- To use a demo, the final entry in the JSON file **must** be a **`user`** message, which serves as the input request for the model.

### Chat Templates

Models in this project support **chat templates** to automatically transform chat-like instructions into structured data for processing.

## Additional Notes

- **Model Support**:
    - Models behind gated access (e.g., LLaMA) require prior approval via the Hugging Face model repository.
    - Refer to the specific demo script for details on model compatibility and features.

- **Errors**: For `Torch not compiled with ...` or similar issues, ensure PyTorch is installed for your system configuration using [this guide](https://pytorch.org/get-started/locally/).

## Further resources

[huggingface model database](https://huggingface.co/models)
