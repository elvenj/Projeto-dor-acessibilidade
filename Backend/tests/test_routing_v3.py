import sys
import unittest

from unittest.mock import MagicMock, patch
from test_rag import dependency, load_module


class RoutingV3Tests(unittest.TestCase):

    def setUp(self):

        self.generate = MagicMock(
            return_value='Resposta do coordenador'
        )

        self.rag = MagicMock(
            return_value='Resposta do especialista'
        )

        self.model = {'model': object()}

        tools = load_module(
            'routing_tools_v3',
            'Services/tools/agent_tools.py',
            {}
        )

        self.coordinator = load_module(
            'routing_coordinator_v3',
            'Agents/coordinatorAgent.py',
            {
                'Services.agents.specialist_agent':
                    dependency(
                        'engine',
                        generate_response=self.generate
                    ),

                'Services.tools.agent_tools':
                    dependency(
                        'tools',
                        build_rag_tool=tools.build_rag_tool
                    ),
            }
        )

        patcher = patch.dict(
            sys.modules,
            {
                'Agents.ragAgent': dependency(
                    'rag',
                    rag_agent=self.rag
                )
            }
        )

        patcher.start()
        self.addCleanup(patcher.stop)

    def ask(self, query, **kwargs):

        return self.coordinator.coordinator_agent(
            self.model,
            query,
            **kwargs
        )

    # T01 — Reproducao da falha experimental

    def test_t02_routes_directly(self):

        query = (
            'Como funciona o vestibular da PUC-Campinas?'
        )

        self.assertEqual(
            self.ask(query),
            'Resposta do especialista'
        )

        self.rag.assert_called_once_with(
            self.model,
            query
        )

        self.generate.assert_not_called()

    # T02 — Normalizacao e variacoes linguisticas

    def test_variations_with_accents(self):

        queries = [
            'Como faço minha MATRÍCULA?',
            'Quais bibliotecas existem?',
            'SBI?'
        ]

        for query in queries:

            with self.subTest(query=query):

                self.rag.reset_mock()

                self.ask(query)

                self.rag.assert_called_once_with(
                    self.model,
                    query
                )

        self.generate.assert_not_called()

    # T03 — Preservacao do contexto

    def test_history_and_url_preserved(self):

        history = [
            {
                'role': 'user',
                'content': 'Quero estudar'
            }
        ]

        url = 'https://www.puc-campinas.edu.br/'

        self.ask(
            ' Vestibular? ',
            history=history,
            current_url=url
        )

        self.rag.assert_called_once_with(
            self.model,
            'Vestibular?',
            history=history,
            current_url=url
        )

    # T04 — Preservacao do fast-path

    def test_greeting_preserved(self):

        self.assertIn(
            'Olá!',
            self.ask('Olá!')
        )

        self.rag.assert_not_called()
        self.generate.assert_not_called()

    # T05 — Intencao desconhecida

    def test_unknown_intent_preserves_llm_route(self):

        self.assertEqual(
            self.ask('Pergunta'),
            'Resposta do coordenador'
        )

        self.generate.assert_called_once()
        self.rag.assert_not_called()

    # T06 — Ferramentas externas

    def test_external_tools_preserve_llm_route(self):

        def consultar_ibge(query: str) -> str:
            return query

        self.ask(
            'Qual é o código IBGE de Campinas?',
            extra_tools=[consultar_ibge]
        )

        self.assertIn(
            consultar_ibge,
            self.generate.call_args.kwargs['tools']
        )

        self.rag.assert_not_called()

    # T07 — Ferramentas concorrentes

    def test_mixed_tools_preserve_llm_route(self):

        def consultar_ibge(query: str) -> str:
            return query

        self.ask(
            'Vestibular da PUC-Campinas?',
            extra_tools=[consultar_ibge]
        )

        self.generate.assert_called_once()
        self.rag.assert_not_called()

    # T08 — Limite de chamadas igual a zero

    def test_zero_budget_preserved(self):

        self.ask(
            'Vestibular?',
            max_iterations=0
        )

        self.assertEqual(
            self.generate.call_args.kwargs['max_iterations'],
            0
        )

        self.rag.assert_not_called()

    # T09 — Limite de chamadas invalido

    def test_negative_budget_rejected(self):

        with self.assertRaises(ValueError):

            self.ask(
                'Vestibular?',
                max_iterations=-1
            )

        self.rag.assert_not_called()
        self.generate.assert_not_called()

    # T10 — Tratamento seguro de falhas

    def test_failure_is_not_exposed(self):

        self.rag.side_effect = RuntimeError(
            'internal details'
        )

        with self.assertLogs(
            self.coordinator.logger,
            level='ERROR'
        ):

            response = self.ask('Vestibular?')

        self.assertIn(
            'Não foi possível',
            response
        )

        self.assertNotIn(
            'internal details',
            response
        )

        self.generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
