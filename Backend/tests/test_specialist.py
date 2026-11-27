import json
import os
import unittest
from unittest.mock import MagicMock, patch

from test_rag import ROOT, dependency, load_module

BASE = 'https://www.puc-campinas.edu.br/'
TARGET = BASE + 'graduacao/sistemas-de-informacao/'


class SpecialistToolsTests(unittest.TestCase):
    def setUp(self):
        self.generate = MagicMock()
        self.rag = load_module('single_specialist', 'Agents/ragAgent.py', {
            'Services.agents.specialist_agent': dependency('engine', generate_response=self.generate),
        })
        self.search = MagicMock(return_value=[{'source': 'manual.pdf', 'page': 42, 'text': 'Há acervo em Braille.'}])
        self.rag.search_evidence = self.search
        graph_module = load_module('graph_tools', 'Services/tools/neo4j_tools.py', {})
        self.service = MagicMock()
        self.rag.build_graph_tools = lambda url, on_result: graph_module.build_neo4j_tools(
            url, service_factory=lambda: self.service, on_result=on_result)

    def tool(self, kwargs, name):
        return next(tool for tool in kwargs['tools'] if tool.__name__ == name)

    def test_specialist_can_choose_documents_even_with_graph_configured(self):
        def choose(*args, **kwargs):
            self.tool(kwargs, 'search_documents')(query='SBI Braille')
            return 'A SBI oferece acervo em Braille. [1]'
        self.generate.side_effect = choose
        with patch.dict(os.environ, {'NEO4J_URI': 'test', 'NEO4J_USER': 'test', 'NEO4J_PASSWORD': 'test'}):
            result = self.rag.rag_agent({}, 'SBI Braille?', current_url=BASE)
        self.assertEqual(result, 'A SBI oferece acervo em Braille. [1]')
        self.service.search_pages.assert_not_called()
        self.service.read_page.assert_not_called()
        self.generate.assert_called_once()

    def test_specialist_can_choose_graph_without_pdf_retrieval(self):
        self.search.return_value = []
        self.service.read_page.return_value = {'url': TARGET, 'title': 'SI', 'text': 'Curso de SI'}
        def choose(*args, **kwargs):
            self.tool(kwargs, 'read_site_page')(url=TARGET)
            return f'Página do curso: {TARGET}'
        self.generate.side_effect = choose
        self.assertIn(TARGET, self.rag.rag_agent({}, 'Onde está o curso?'))
        self.search.assert_called_once_with('Onde está o curso?', top_k=5)
        self.service.read_page.assert_called_once_with(url=TARGET)

    def test_specialist_can_combine_pdf_and_navigation_tools(self):
        self.service.navigation_path.return_value = {'pages': [{'url': BASE}, {'url': TARGET}], 'link_texts': [''], 'click_count': 1}
        def choose(*args, **kwargs):
            self.tool(kwargs, 'search_documents')(query='Braille')
            self.tool(kwargs, 'find_site_path')(target_url=TARGET)
            return f'Há acervo em Braille [1]. Caminho: {BASE} → {TARGET}'
        self.generate.side_effect = choose
        history = [{'role': 'user', 'content': 'SBI'}]
        answer = self.rag.rag_agent({}, 'Como chego?', current_url=BASE, history=history)
        self.assertNotIn('manual.pdf', answer)
        self.assertIn('[1]', answer)
        self.assertIn(TARGET, answer)
        self.service.navigation_path.assert_called_once_with(start_url=BASE, target_url=TARGET)
        kwargs = self.generate.call_args.kwargs
        self.assertEqual(kwargs['history'], history)
        self.assertEqual(json.loads(kwargs['prompt'])['current_url'], BASE)
        self.assertNotIn('fallback_tool', kwargs)
        self.assertNotIn('return_direct_tools', kwargs)

    def test_initial_evidence_is_delivered_without_requiring_tool_selection(self):
        self.generate.return_value = 'A SBI oferece livros em Braille.'
        answer = self.rag.rag_agent({}, 'SBI?')
        self.assertEqual(answer, 'A SBI oferece livros em Braille.')
        self.search.assert_called_once_with('SBI?', top_k=5)
        payload = json.loads(self.generate.call_args.kwargs['prompt'])
        self.assertIn('Há acervo em Braille.', payload['documents'][0]['text'])
        self.service.search_pages.assert_not_called()
        self.service.read_page.assert_not_called()

    def test_no_evidence_blocks_unsupported_answer(self):
        self.search.return_value = []
        self.generate.return_value = 'Resposta da LLM sem evidência.'
        self.assertEqual(
            self.rag.rag_agent({}, 'SBI?'),
            self.rag.INSUFFICIENT_EVIDENCE
        )

    def test_refusal_with_context_is_returned_without_extractive_replacement(self):
        texts = json.loads((ROOT / 'tests/fixtures/area_logada.json').read_text(encoding='utf-8-sig'))
        self.search.return_value = [{'source': 'guia.pdf', 'page': 14, 'text': text} for text in texts]
        self.generate.return_value = 'Não encontrei informações suficientes para responder.'
        self.assertEqual(self.rag.rag_agent({}, 'Como funciona a area logada?'), self.generate.return_value)
        payload = json.loads(self.generate.call_args.kwargs['prompt'])
        self.assertEqual(payload['documents'][0]['text'], texts[0])

    def test_refusal_after_empty_graph_is_not_overridden(self):
        self.service.search_pages.return_value = []
        def choose(*args, **kwargs):
            self.tool(kwargs, 'search_site_pages')(query='biblioteca')
            return 'Não encontrei informações.'
        self.generate.side_effect = choose
        self.assertEqual(self.rag.rag_agent({}, 'Biblioteca?'), 'Não encontrei informações.')

    def test_tool_failure_without_evidence_blocks_answer(self):
        self.search.side_effect = RuntimeError('failed')
        self.generate.return_value = 'Posso explicar mesmo assim.'
        with self.assertLogs(self.rag.build_document_tool.__module__, level='ERROR'):
            self.assertEqual(
                self.rag.rag_agent({}, 'Pergunta'),
                self.rag.INSUFFICIENT_EVIDENCE
            )

    def test_empty_graph_and_documents_block_answer(self):

        self.search.return_value = []
        self.service.search_pages.return_value = []

        def choose(*args, **kwargs):

            self.tool(
                kwargs, 'search_site_pages'
            )(query='biblioteca')

            return 'Resposta sem fundamentação.'

        self.generate.side_effect = choose

        self.assertEqual(
            self.rag.rag_agent({}, 'Biblioteca?'),
            self.rag.INSUFFICIENT_EVIDENCE
        )

    def test_unconfirmed_url_is_blocked(self):
        self.generate.return_value = 'Veja https://inventado.example/ [99]'
        self.assertEqual(
            self.rag.rag_agent({}, 'curso SI'),
            self.rag.UNVERIFIED_URL_RESPONSE
        )
        self.service.search_pages.assert_not_called()

    def test_verified_document_url_is_preserved(self):

        self.search.return_value = [{
            "source": "manual.pdf",
            "page": 1,
            "text": "Acesse " + TARGET
        }]

        self.generate.return_value = "Acesse " + TARGET

        self.assertEqual(
            self.rag.rag_agent({}, "Onde acesso o curso?"),
            self.generate.return_value
        )

    def test_mixed_verified_and_unverified_urls_are_blocked(self):

        self.search.return_value = [{
            "source": "manual.pdf",
            "page": 1,
            "text": "Acesse " + TARGET
        }]

        self.generate.return_value = (
            TARGET + " ou " + "https:" + "//inventado.example/"
        )

        self.assertEqual(
            self.rag.rag_agent({}, "Onde acesso o curso?"),
            self.rag.UNVERIFIED_URL_RESPONSE
        )

    def test_extracts_real_document_steps(self):

        document = json.loads(
            (ROOT / "tests/fixtures/vestibular_steps.json")
            .read_text(encoding="utf-8")
        )

        self.search.return_value = [document]

        answer = self.rag.rag_agent(
            {},
            "Quais são as quatro etapas do formulário de inscrição?"
        )

        for term in ["Login", "Curso", "Dados", "Inscrição"]:
            self.assertIn(term, answer)

        self.generate.assert_not_called()

    def test_responses_are_returned_verbatim(self):
        responses = [
            '',
            'Não sei responder.',
            'Não tenho informação suficiente.',
            'Veja https://nao-consultado.example/ [123]',
            'Texto.\n\nFontes:\nmanual.pdf, página 99',
            '  Resposta com espaços e\nquebra de linha.  ',
        ]
        for response in responses:
            with self.subTest(response=response):
                self.generate.return_value = response
                expected = (
                    self.rag.UNVERIFIED_URL_RESPONSE
                    if response.startswith('Veja ')
                    else response
                )

                self.assertEqual(
                    self.rag.rag_agent({}, 'Pergunta'),
                    expected
                )


if __name__ == '__main__':
    unittest.main()
