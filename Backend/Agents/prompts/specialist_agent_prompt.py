
"""Instruções compartilhadas pelo executor de ferramentas — V2."""


INVALID_TOOL_CALL_PROMPT = (
    'A chamada anterior é inválida.\n'
    'Corrija a chamada seguindo estas regras:\n'
    '1. Selecione uma ferramenta disponível.\n'
    '2. Utilize o nome exato da ferramenta.\n'
    '3. Forneça os argumentos exigidos pela ferramenta.\n'
    '4. Utilize JSON válido.\n'
    '5. Retorne somente a chamada corrigida.\n\n'
    'Formato obrigatório:\n'
    '<tool_call>{"name": "nome", "arguments": {}}</tool_call>\n\n'
    'Substitua o nome e os argumentos pelos valores corretos.\n'
    'Preserve a intenção original do usuário.\n'
    'Não invente ferramentas, argumentos ou resultados.'
)
