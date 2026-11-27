import logging
import re


UNVERIFIED_URL_RESPONSE = (
    "Não consegui confirmar o endereço solicitado. "
    "Prefiro não indicar um link sem confirmação."
)

URL_PATTERN = re.compile(
    r"https?://[^\s<>\[\]\"'()]+",
    re.IGNORECASE
)


def extract_urls(value):

    if isinstance(value, str):

        return {
            url.rstrip(".,;:!?")
            for url in URL_PATTERN.findall(value)
        }

    if isinstance(value, dict):

        return extract_urls(list(value.values()))

    if isinstance(value, (list, tuple)):

        urls = set()

        for item in value:
            urls.update(extract_urls(item))

        return urls

    return set()


def verified_urls(documents, observations=()):

    allowed = set()

    for document in documents:

        allowed.update(
            extract_urls(document.get("text", ""))
        )

    for operation, result in observations:

        if (
            operation != "search_documents"
            and isinstance(result, dict)
            and result.get("status") == "ok"
        ):

            allowed.update(
                extract_urls(result.get("data"))
            )

    return allowed


def has_unverified_urls(response, allowed):

    return bool(
        extract_urls(response) - allowed
    )


INSUFFICIENT_EVIDENCE = (
    "Não encontrei informações suficientes para responder."
)


def extract_numbered_steps(query, documents):

    import unicodedata

    normalized = "".join(
        c for c in unicodedata.normalize(
            "NFKD", query.casefold()
        )
        if not unicodedata.combining(c)
    )

    numbers = {
        "duas": 2, "dois": 2,
        "tres": 3, "quatro": 4,
        "cinco": 5, "seis": 6,
        "sete": 7, "oito": 8,
        "nove": 9, "dez": 10
    }

    pattern = (
        r"\b(\d+|duas|dois|tres|quatro|cinco|"
        r"seis|sete|oito|nove|dez)\s+etapas\b"
    )

    match = re.search(pattern, normalized)

    if not match:
        return None

    value = match.group(1)

    count = (
        int(value)
        if value.isdigit()
        else numbers[value]
    )

    if not 2 <= count <= 20:
        return None

    for document in documents:

        text = document.get("text", "")

        steps = re.findall(
            r"\bETAPA\s+(\d+)\s*[-–—:]\s*([^\n;]+)",
            text,
            flags=re.IGNORECASE
        )

        indices = [int(number) for number, _ in steps]

        if indices != list(range(1, count + 1)):
            continue

        answer = [
            f"{number}. {description.strip().rstrip('.;')}."
            for number, description in steps
        ]

        logging.getLogger(__name__).info(
            "generation=extractive source=%s page=%s",
            document.get("source"),
            document.get("page")
        )

        return "\n".join(answer)

    return None


def has_document_evidence(documents):

    return any(
        isinstance(item, dict)
        and isinstance(item.get("text"), str)
        and item["text"].strip()
        for item in documents
    )


def has_graph_evidence(observations):

    return any(
        operation != "search_documents"
        and isinstance(result, dict)
        and result.get("status") == "ok"
        and bool(result.get("data"))
        for operation, result in observations
    )


from Services.agents.specialist_agent import generate_response
from Services.tools.document_tools import build_document_tool
from Agents.prompts.ragAgent_prompt import SPECIALIST_PROMPT, build_user_prompt as build_specialist_prompt
from Agents.prompts.ragAgent_diagnostic_prompt import SYSTEM_PROMPT, build_user_prompt as build_diagnostic_prompt


def search_evidence(query: str, top_k: int = 5):
    from Services.orm.search import search_evidence as search
    return search(query, top_k=top_k)


def diagnose_rag(llm, query: str, evidence: list[dict] | None = None, *, history: list | None = None):
    if evidence is None:
        evidence = search_evidence(query, top_k=5)
    report = {"query": query, "evidence": evidence, "raw_response": None,
              "status": "pending", "response": None}
    if not has_document_evidence(evidence):

        report.update(
            status="insufficient_evidence",
            response=INSUFFICIENT_EVIDENCE
        )

        return report

    response = generate_response(
        llm=llm,
        prompt=build_diagnostic_prompt(query, evidence),
        tools=[],
        system_prompt=SYSTEM_PROMPT,
        history=history,
    )
    if has_unverified_urls(
        response, verified_urls(evidence)
    ):

        report.update(
            raw_response=response,
            response=UNVERIFIED_URL_RESPONSE,
            status="unverified_url"
        )

        return report

    report.update(raw_response=response, response=response,
                  status='answered' if response.strip() else 'empty_generation')
    return report


def build_graph_tools(current_url, on_result):
    from Services.tools.neo4j_tools import build_neo4j_tools
    return build_neo4j_tools(current_url, on_result=on_result)


def rag_agent(llm, query: str, *, history: list | None = None, current_url: str | None = None):

    observations = []
    documents = []

    search_documents = build_document_tool(documents, search=search_evidence,
        on_result=lambda operation, result: observations.append((operation, result)))

    graph_tools = build_graph_tools(current_url,
        on_result=lambda operation, result: observations.append((operation, result)))
    search_documents(query=query)

    extracted = extract_numbered_steps(query, documents)

    if extracted is not None:
        return extracted
    prompt = build_specialist_prompt(query, current_url, documents=documents)
    logging.getLogger(__name__).info('Especialista: documentos iniciais=%s', len(documents))
    response = generate_response(llm, prompt=prompt,
        tools=[search_documents, *graph_tools], system_prompt=SPECIALIST_PROMPT,
        history=history, max_iterations=4)

    if not (
        has_document_evidence(documents)
        or has_graph_evidence(observations)
    ):

        logging.getLogger(__name__).warning(
            "evidence_gate=blocked reason=no_evidence"
        )

        return INSUFFICIENT_EVIDENCE

    if has_unverified_urls(
        response,
        verified_urls(documents, observations)
    ):

        logging.getLogger(__name__).warning(
            "output_guardrail=blocked reason=unverified_url"
        )

        return UNVERIFIED_URL_RESPONSE

    return response
