# Standard library
import os
from typing import Union, Generator, Dict, List

# Third party
import google.generativeai as genai
from ollama import chat
from openai import OpenAI
from anthropic import Anthropic


#########
# HELPERS
#########


MAX_TOKENS = 8192


def get_llm_provider():
    if os.getenv("OPENAI_API_KEY", None):  # Default
        return "openai"

    if os.getenv("ANTHROPIC_API_KEY", None):
        return "anthropic"

    if os.getenv("GEMINI_API_KEY", None):
        return "gemini"

    if os.getenv("OLLAMA_MODEL", None):
        return "ollama"

    raise ValueError(
        "No API key found for OpenAI, Anthropic, or Gemini. No Ollama model found either."
    )


def call_openai(
    system: str, messages: List[Dict[str, str]], **kwargs
) -> Union[str, Generator[str, None, None]]:
    openai = OpenAI()
    stream = False if "stream" not in kwargs else kwargs["stream"]
    response = openai.chat.completions.create(
        messages=[{"role": "system", "content": system}, *messages],
        model="gpt-4o" if "model" not in kwargs else kwargs["model"],
        temperature=0.7 if "temperature" not in kwargs else kwargs["temperature"],
        stream=stream,
        max_tokens=MAX_TOKENS,
    )

    if not stream:
        return response.choices[0].message.content

    response = (e.choices[0] for e in response)
    response = (e for e in response if e.finish_reason != "stop" and e.delta.content)
    response = (e.delta.content for e in response)

    return response


def call_anthropic(
    system: str, messages: List[Dict[str, str]], **kwargs
) -> Union[str, Generator[str, None, None]]:
    anthropic = Anthropic()
    stream = False if "stream" not in kwargs else kwargs["stream"]
    response = anthropic.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=MAX_TOKENS,
        system=system,
        messages=messages,
        temperature=0.7 if "temperature" not in kwargs else kwargs["temperature"],
        stream=stream,
    )

    if not stream:
        return response.content[0].text

    response = (e for e in response if e.type == "content_block_delta")
    response = (e.delta.text for e in response)

    return response


def call_ollama(
    system: str, messages: List[Dict[str, str]], **kwargs
) -> Union[str, Generator[str, None, None]]:
    stream = kwargs.get("stream", False)

    if not stream:
        response = chat(
            model=os.getenv("OLLAMA_MODEL", None),
            messages=[{"role": "system", "content": system}, *messages],
        )
        return response.message.content
    else:
        response = chat(
            model=os.getenv("OLLAMA_MODEL", None),
            messages=[{"role": "system", "content": system}, *messages],
            stream=True,
        )

        def stream_generator():
            for chunk in response:
                if chunk and "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]

        return stream_generator()


def call_gemini(
    system: str, messages: List[Dict[str, str]], **kwargs
) -> Union[str, Generator[str, None, None]]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    genai.configure(api_key=api_key)

    model_name = kwargs.get("model", "gemini-pro")
    temperature = kwargs.get("temperature", 0.7)

    # genai.types.GenerationConfig is the correct way for older versions,
    # but for newer ones (like 0.5.0+), it's genai.GenerationConfig
    # Using a try-except block for broader compatibility or checking version
    # For now, assuming a version where genai.GenerationConfig is available
    try:
        generation_config = genai.GenerationConfig(temperature=temperature)
    except AttributeError:
        # Fallback for older versions if genai.types.GenerationConfig was the path
        # This is a common pattern if library structures change.
        # However, the problem description implies genai.types.GenerationConfig
        # was what the user might have seen. Let's stick to what was implied unless errors occur.
        # For the pyright ignore, it suggests genai.types might be an issue.
        # Let's assume genai.GenerationConfig is the more modern/correct one.
        generation_config = genai.types.GenerationConfig(temperature=temperature) # pyright: ignore[reportAttributeAccessIssue]


    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        system_instruction=system,
    )

    mapped_messages = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role.lower() in ["assistant", "bot", "model"]: # model is already the target role
            mapped_role = "model"
        elif role.lower() == "user":
            mapped_role = "user"
        else:
            # Skip system messages as they are handled by system_instruction
            # and unknown roles.
            continue
        mapped_messages.append({"role": mapped_role, "parts": [content]})

    # Ensure messages are not empty after mapping, Gemini requires non-empty messages
    if not mapped_messages:
        # This can happen if only system messages were passed and they are now in system_instruction
        # Or if all messages had roles that were filtered out.
        # Gemini API requires at least one message.
        # Depending on desired behavior, could raise error or return empty/default response.
        # For now, let's assume messages will contain at least one valid user/model message.
        pass


    stream = kwargs.get("stream", False)

    try:
        if not stream:
            response = model.generate_content(mapped_messages)
            return response.text
        else:
            response = model.generate_content(mapped_messages, stream=True)

            def stream_generator():
                for chunk in response:
                    # Add check for chunk.text, as sometimes empty chunks might appear
                    if hasattr(chunk, 'text'):
                        yield chunk.text
            return stream_generator()
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        # Consider specific exceptions like google.api_core.exceptions.GoogleAPIError
        # For now, re-raising the caught exception.
        raise


######
# MAIN
######


def call_llm(
    system: str, messages: List[Dict[str, str]], **kwargs
) -> Union[str, Generator[str, None, None]]:
    provider = get_llm_provider()
    if provider == "openai":
        return call_openai(system, messages, **kwargs)
    elif provider == "anthropic":
        return call_anthropic(system, messages, **kwargs)
    elif provider == "ollama":
        return call_ollama(system, messages, **kwargs)
    elif provider == "gemini":
        return call_gemini(system, messages, **kwargs)
