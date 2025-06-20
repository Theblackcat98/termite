# 🐛 Termite

Generate terminal UIs (TUIs) with simple text prompts.

![Demo](assets/demo.gif)

Termite lets you quickly prototype a terminal app to solve a problem. It works well for tasks like:

- "Show me which ports are active"
- "Make me a throughput monitor for my Redis queue"
- "Help me manage my Docker containers"
- "Diff these two SQL tables"

Under the hood, an LLM is generating and auto-executing a Python script that implements the UI. By default, UIs are built with the [urwid](https://urwid.org/) library, but you can also use [rich](https://rich.readthedocs.io/en/latest/), [curses](https://docs.python.org/3/library/curses.html), or [textual](https://textual.textualize.io/).

Please use with caution. Termite is still very experimental and it's obviously risky to run AI-generated code.

## Installation

```bash
> pipx install termite-ai
```

Once installed, you can use OpenAI or Anthropic as your LLM provider. Just add the appropriate API key to your environment:

```bash
> export OPENAI_API_KEY="..." # For GPT-4o, etc.
> export ANTHROPIC_API_KEY="..." # For Claude Sonnet, etc.
> export GEMINI_API_KEY="..." # For Gemini Pro, etc.
> export OLLAMA_MODEL="..." # For locally hosted models like Llama 3, e.g., OLLAMA_MODEL="llama3"
```

If you're using OpenAI, you can also set your API URL by adding the following to your environment:

```bash
> export OPENAI_BASE_URL="..." # Defaults to None
```

**Provider Details:**

*   **OpenAI & Anthropic**: These are the recommended providers for best results.
*   **Gemini**: Uses the `google-generativeai` Python package (which will be installed as a dependency if you use Gemini). It requires the `GEMINI_API_KEY` to be set. The default model used is `gemini-pro`.
*   **Ollama**: Allows you to use locally hosted LLMs. You need to set the `OLLAMA_MODEL` environment variable to specify which model Ollama should use (e.g., `export OLLAMA_MODEL="llama3"`). Ollama integration supports streaming responses.

**Provider Priority:**

Termite will select an LLM provider based on the environment variables you have set, in the following order of priority:
1.  OpenAI (`OPENAI_API_KEY`)
2.  Anthropic (`ANTHROPIC_API_KEY`)
3.  Gemini (`GEMINI_API_KEY`)
4.  Ollama (`OLLAMA_MODEL`)

For example, if you have both `OPENAI_API_KEY` and `GEMINI_API_KEY` set, Termite will use OpenAI.

## Running without installation

If you prefer not to install Termite globally, you can run it directly from the cloned repository:

1.  Clone the repository:
    ```bash
    git clone https://github.com/shobrook/termite.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd termite
    ```
3.  Install dependencies:
    ```bash
    pip install rich requests google-generativeai ollama openai anthropic urwid textual
    ```
4.  Set the PYTHONPATH to include the current directory. This allows Python to find the Termite modules:
    ```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    ```
5.  Run the tool using the Python module execution flag `-m`:
    ```bash
    python -m termite --help
    ```
    You can replace `--help` with any other Termite commands or prompts.

## Usage

To use, run the following:

```bash
> termite
```

You'll be asked to describe what you want to build. Do this, and then Termite will execute the following steps:

1. Generate a design document based on your prompt.
2. Implement the TUI in Python.
3. Iteratively fix runtime errors, if any exist.
4. (Optional) Iteratively refine the TUI based on self-reflections.

Once finished, your TUI will be saved to the `~/.termite` directory and automatically started up for you to use.

### Advanced Usage

You can configure the steps above with the following command-line arguments:

- `--library`: Specify the library Termite should use to build the TUI. Options are: urwid, rich, textual, and curses. Default is urwid.
- `--refine`: Setting this will improve the output by adding a self-reflection and refinement step to the process.
- `--refine-iters`: Controls the number of times the TUI should be refined, if `--refine` is enabled. Default is 1.
- `--fix-iters`: Controls the maximum number of attempts Termite should make at fixing a bug with the TUI. Default is 10.

## Examples

> Built something cool? [Submit a PR](https://github.com/shobrook/termite/pulls) to add your example here.

**"Make me a simple process monitor"**

![Process](./assets/process.png)

**"Help me manage my Git branches (view, create, switch, etc.)"**

![Git](./assets/git.png)

**"I need a quick way to test some RegEx patterns"**

![RegEx](./assets/regex.png)

## Roadmap

The bottleneck in most code generation pipelines is the verifier. That is, how can we verify that the generated code is what we want? Termite currently does the bare minimum for verification –– we execute the TUI in a pseudo-terminal to detect runtime exceptions. But a TUI can still look or behave improperly even if it runs without errors...

Some ideas:

1. Capture an image of what the TUI looks like and feed it to GPT-4o Vision for evaluation. This could help verify UI correctness.
2. Use an agent to simulate user actions in the TUI and record the results. This could help verify functionality correctness. But the problem here is some actions in a TUI can be destructive (e.g. killing a process in `htop`).
