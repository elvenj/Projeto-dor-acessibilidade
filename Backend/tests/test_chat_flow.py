"""Regressão do fluxo controller -> coordenador -> especialista -> recuperação."""
import json
import sys
import unittest
from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

from test_rag import ROOT, dependency, load_module


class ChatFlowTests(unittest.TestCase):
    def test_direct_institutional_pipeline(self):
        self.check_pipeline(direct=True)

    def test_legacy_llm_pipeline(self):
        self.check_pipeline(direct=False)

    def check_pipeline(self, direct):
        query = 'Como funciona a area logada da puc-campinas?'
        texts = json.loads((ROOT / 'tests/fixtures/area_logada.json').read_text(encoding='utf-8-sig'))
        engine = load_module('flow_engine', 'Services/agents/specialist_agent.py', {'torch': dependency('torch')})
        turns = [
            '<tool_call>{"name":"consult_knowledge_base","arguments":{"query":"pergunta completa"}}</tool_call>',
            'Não encontrei informações suficientes para responder.',
            'Resposta final do coordenador. https://example.com/ [9]',
        ]

        if direct:
            turns = turns[1:2]

        engine.generate_turn = MagicMock(side_effect=turns)

        with patch.object(sys, 'path', [str(ROOT), *sys.path]), patch.dict(sys.modules, {
            'Services.agents.specialist_agent': dependency('engine', generate_response=engine.generate_response),
        }):
            rag = load_module('flow_rag', 'Agents/ragAgent.py', {})
            rag.search_evidence = MagicMock(return_value=[{'text': text, 'source': 'guia.pdf', 'page': 14} for text in texts])
            rag.build_graph_tools = lambda *args, **kwargs: []
            coordinator = load_module('flow_coordinator', 'Agents/coordinatorAgent.py', {})
            with patch.dict(sys.modules, {
                'Agents.ragAgent': rag,
                'Agents.coordinatorAgent': coordinator,
            }):
                controller = load_module('flow_controller', 'Controllers/LLM/coordinatorController.py', {})
                store = MagicMock()
                chat_id = '90c36ab5-1ae3-4c3e-96eb-bc0b7d6da8b5'
                store.create_chat.return_value = chat_id
                store.snapshot.return_value = ([], 0)
                request = SimpleNamespace(client=SimpleNamespace(host='127.0.0.1'),
                                          app=SimpleNamespace(state=SimpleNamespace(llm={})))
                routing = (
                    nullcontext() if direct
                    else patch.object(
                        coordinator,
                        'is_institutional',
                        return_value=False
                    )
                )

                with routing, patch.object(controller, 'ChatStore', return_value=store), patch.object(controller, 'encrypted', return_value='hash-ip'):
                    answer = controller.chat(controller.ChatRequest(search=query), request)
                self.assertEqual(answer['chat_id'], chat_id)
                self.assertEqual(answer['query'], query)
                self.assertEqual(answer['response'], turns[-1])
                rag.search_evidence.assert_called_once_with(query, top_k=5)
                self.assertEqual(engine.generate_turn.call_count, 1 if direct else 3)
                store.save_turn.assert_called_once_with('hash-ip', chat_id, query, answer['response'], expected_version=0)


if __name__ == '__main__':
    unittest.main()
