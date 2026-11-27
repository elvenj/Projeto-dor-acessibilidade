import inspect
import json
import logging
import re

import torch
from Agents.prompts.specialist_agent_prompt import INVALID_TOOL_CALL_PROMPT

logger = logging.getLogger(__name__)
TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def generate_turn(llm, messages: list, tools: list) -> str:
    model, tokenizer = llm['model'], llm['tokenizer']
    text = tokenizer.apply_chat_template(
        messages, **({'tools': tools} if tools else {}),
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=llm.get('max_new_tokens', 512), do_sample=False)
    generated = outputs[0][inputs['input_ids'].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=not tools).strip()


def generate_response(llm, prompt: str, tools: list, max_iterations: int = 3,
                      system_prompt: str | None = None, history: list | None = None):
    if max_iterations < 0:
        raise ValueError('max_iterations não pode ser negativo.')
    tool_map = {tool.__name__: tool for tool in tools}
    if len(tool_map) != len(tools):
        raise ValueError('Os nomes das ferramentas devem ser únicos.')
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    for message in (history or [])[-20:]:
        if message.get('role') in {'user', 'assistant'} and isinstance(message.get('content'), str):
            messages.append({'role': message['role'], 'content': message['content'][:2000]})
    messages.append({'role': 'user', 'content': prompt})
    remaining = max_iterations

    for _ in range(max_iterations + 1):
        response = generate_turn(llm, messages, tools if remaining else [])
        matches = TOOL_CALL.findall(response)
        if not matches and tools:
            # Alguns modelos retornam JSON válido sem o delimitador <tool_call>.
            candidate = re.sub(r'<\|im_end\|>\s*$', '', response.strip())
            candidate = re.sub(r'^```(?:json)?\s*|\s*```$', '', candidate)
            try:
                parsed = json.loads(candidate)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, dict) and 'name' in parsed and 'arguments' in parsed:
                matches = [candidate]
        if not matches:
            if "<tool_call>" in response or "</tool_call>" in response:
                return "Não consegui concluir a consulta."
            return re.sub(r'<\|im_end\|>\s*$', '', response).strip()
        if not remaining:
            return (
                "Não consegui concluir a consulta. "
                "Por favor, reformule sua pergunta."
            )

        calls = []
        try:
            for match in matches:
                function = json.loads(match)
                if not isinstance(function, dict) or not isinstance(function.get('name'), str):
                    raise ValueError('Chamada deve conter name e arguments.')
                if not isinstance(function.get('arguments'), dict):
                    raise ValueError('arguments deve ser um objeto JSON.')
                calls.append({'type': 'function', 'function': function})
        except (ValueError, TypeError):
            remaining -= 1
            messages.extend([
                {'role': 'assistant', 'content': response},
                {'role': 'user', 'content': INVALID_TOOL_CALL_PROMPT},
            ])
            continue

        messages.append({'role': 'assistant', 'content': '', 'tool_calls': calls})
        for call in calls:
            function = call['function']
            name, arguments = function['name'], function['arguments']
            if remaining <= 0:
                result = 'Ferramenta não executada: limite de chamadas atingido.'
            else:
                remaining -= 1
                if name not in tool_map:
                    result = 'Ferramenta não disponível. Use apenas as ferramentas listadas.'
                else:
                    tool = tool_map[name]
                    try:
                        inspect.signature(tool).bind(**arguments)
                    except TypeError:
                        result = 'Argumentos inválidos. Confira os campos exigidos pela ferramenta.'
                    else:
                        try:
                            result = tool(**arguments)
                        except Exception:
                            logger.exception('Falha na ferramenta %s', name)
                            result = 'Erro ao executar a ferramenta. Não foi possível concluir a consulta.'
            messages.append({'role': 'tool', 'name': name, 'content': str(result)})
    return response
