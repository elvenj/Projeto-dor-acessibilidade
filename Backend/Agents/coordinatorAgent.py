import logging
import re
import unicodedata

from Services.agents.specialist_agent import generate_response
from Services.tools.agent_tools import build_rag_tool
from Agents.prompts.coordinatorAgent_prompt import SYSTEM_PROMPT, get_system_prompt

logger = logging.getLogger(__name__)


# Vocabulário institucional inicial

INSTITUTIONAL_TERMS = re.compile(
    r"\b(?:puc[\s-]*campinas|sbi|vestibular(?:es)?|"
    r"matriculas?|rematriculas?|bibliotecas?|"
    r"prouni|bolsas? de estudo|area logada|portal do aluno)\b"
)


def is_institutional(query: str) -> bool:
    """Identifica consultas institucionais por regras lexicais."""

    normalized = unicodedata.normalize(
        "NFKD", query.casefold()
    )

    normalized = "".join(
        c for c in normalized
        if not unicodedata.combining(c)
    )

    return bool(
        INSTITUTIONAL_TERMS.search(normalized)
    )


def coordinator_agent(
    llm,
    query: str,
    *,
    extra_tools: list | None = None,
    max_iterations: int = 3,
    history: list | None = None,
    current_url: str | None = None
):

    # Validação da entrada

    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            'A pergunta não pode estar vazia.'
        )

    query = query.strip()

    # Fast-path para saudações

    if re.fullmatch(
        r'(oi|olá|ola|bom dia|boa tarde|boa noite|obrigado|obrigada)[!.,\s]*',
        query,
        re.I
    ):
        return (
            'Olá! Posso ajudar com informações e serviços '
            'da PUC-Campinas. O que você precisa saber?'
        )

    if max_iterations < 0:
        raise ValueError(
            'max_iterations não pode ser negativo.'
        )

    # Preservação do contexto

    context = {}

    if history:
        context['history'] = history

    if current_url:
        context['current_url'] = current_url

    # Construção da ferramenta RAG existente

    rag_tool = build_rag_tool(
        llm,
        original_query=query,
        **context
    )

    # Roteamento determinístico institucional

    if (
        max_iterations > 0
        and not extra_tools
        and is_institutional(query)
    ):

        logger.info(
            'route=institutional_direct '
            'tool=consult_knowledge_base'
        )

        try:

            return rag_tool(query=query)

        except Exception:

            logger.exception(
                'Falha na consulta institucional direta'
            )

            return (
                'Erro ao executar a ferramenta. '
                'Não foi possível concluir a consulta.'
            )

    # Fallback: seleção de ferramentas pelo LLM

    logger.info('route=llm_tool_selection')

    tools = [
        rag_tool,
        *(extra_tools or [])
    ]

    return generate_response(
        llm=llm,
        prompt=query,
        tools=tools,
        system_prompt=get_system_prompt(
            has_current_url=bool(current_url)
        ),
        max_iterations=max_iterations,
        history=history,
    )
