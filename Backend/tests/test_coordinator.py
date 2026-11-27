import json
import sys
import unittest
from unittest.mock import MagicMock, patch

from test_rag import ROOT, dependency, load_module


class ToolExecutionTests(unittest.TestCase):
    def setUp(self):
        self.engine = load_module('engine_under_test', 'Services/agents/specialist_agent.py', {
            'torch': dependency('torch'),
        })
        self.turn = MagicMock()
        self.engine.generate_turn = self.turn
        self.executed = []

        def lookup(query: str) -> str:
            self.executed.append(query)
            return 'Há livros em Braille. [1] guia.pdf, página 12'
        self.tool = lookup

    def call(self, name='lookup', arguments=None):
        return '<tool_call>' + json.dumps({'name': name, 'arguments': arguments or {'query': 'Braille?'}}) + '</tool_call>'

    def test_direct_response_does_not_execute_tools(self):
        self.turn.return_value = 'Olá! Como posso ajudar?'
        self.assertIn('Olá', self.engine.generate_response({}, 'oi', [self.tool]))
        self.assertEqual(self.executed, [])

    def test_refusal_without_tools_is_returned_without_retry(self):
        self.turn.return_value = 'Não tenho informações.'
        self.assertEqual(self.engine.generate_response({}, 'Braille?', [self.tool]), 'Não tenho informações.')
        self.assertEqual(self.turn.call_count, 1)
        self.assertEqual(self.executed, [])

    def test_json_without_delimiters_is_executed(self):
        self.turn.side_effect = ['```json\n{"name":"lookup","arguments":{"query":"Braille?"}}\n```<|im_end|>', 'Há Braille.']
        self.engine.generate_response({}, 'Braille?', [self.tool])
        self.assertEqual(self.executed, ['Braille?'])

    def test_empty_answer_is_not_replaced(self):
        self.turn.return_value = ''
        self.assertEqual(self.engine.generate_response({}, 'Braille?', [self.tool]), '')
        self.assertEqual(self.turn.call_count, 1)

    def test_history_is_passed_before_current_question(self):
        self.turn.return_value = 'Resposta'
        history = [{'role': 'user', 'content': 'Livros em Braille?'},
                   {'role': 'assistant', 'content': 'Disponíveis na SBI.'}]
        self.engine.generate_response({}, 'E onde?', [], system_prompt='Regras', history=history)
        messages = self.turn.call_args.args[1]
        self.assertEqual(messages[1:3], history)
        self.assertEqual(messages[-1], {'role': 'user', 'content': 'E onde?'})

    def test_unverified_coordinator_answer_is_not_discarded(self):
        self.turn.return_value = 'Yes, SBI means São Bento Institute.'
        self.assertEqual(self.engine.generate_response({}, 'SBI?', [self.tool]), self.turn.return_value)
        self.assertEqual(self.executed, [])

    def test_coordinator_final_generation_is_returned_after_tool_execution(self):
        self.turn.side_effect = [self.call(), 'Resposta final da LLM. https://example.com/']
        answer = self.engine.generate_response({}, 'SBI?', [self.tool])
        self.assertEqual(self.turn.call_count, 2)
        self.assertEqual(answer, 'Resposta final da LLM. https://example.com/')

    def test_tool_result_is_delivered_to_model(self):
        self.turn.side_effect = [self.call(), 'Sim, há Braille. [1] guia.pdf, página 12']
        response = self.engine.generate_response({}, 'Braille?', [self.tool], system_prompt='Regras')
        self.assertEqual(self.executed, ['Braille?'])
        history = self.turn.call_args.args[1]
        self.assertEqual(history[0]['role'], 'system')
        self.assertEqual(history[-1]['role'], 'tool')
        self.assertIn('guia.pdf', history[-1]['content'])
        self.assertIn('guia.pdf', response)

    def test_invalid_unknown_and_wrong_arguments_do_not_crash(self):
        for response in ['<tool_call>invalid</tool_call>', self.call('unknown'),
                         self.call(arguments={'unexpected': 'x'})]:
            self.turn.side_effect = [response, 'Não foi possível consultar.']
            self.engine.generate_response({}, 'teste', [self.tool])
        self.assertEqual(self.executed, [])

    def test_batch_cannot_bypass_call_budget(self):
        self.turn.side_effect = [self.call() * 5, 'Resultado disponível.']
        self.engine.generate_response({}, 'teste', [self.tool], max_iterations=1)
        self.assertEqual(len(self.executed), 1)
        self.assertEqual(self.turn.call_args.args[2], [])

    def test_repeated_tool_calls_stop(self):
        self.turn.return_value = self.call()
        response = self.engine.generate_response({}, 'teste', [self.tool], max_iterations=2)
        self.assertEqual(len(self.executed), 2)
        self.assertNotIn('<tool_call>', response)
        self.assertIn('Não consegui concluir', response)

    def test_tool_exception_is_reported_without_exposing_exception(self):
        def broken(query: str):
            raise RuntimeError('internal details')
        self.turn.side_effect = [self.call('broken'), 'A consulta falhou.']
        with self.assertLogs(self.engine.logger, level='ERROR'):
            self.engine.generate_response({}, 'teste', [broken])
        result = self.turn.call_args.args[1][-1]['content']
        self.assertIn('Erro', result)
        self.assertNotIn('internal details', result)


class CoordinatorTests(unittest.TestCase):
    def test_binds_rag_model_and_registers_extra_tool(self):
        generate = MagicMock(return_value='Resposta')
        rag = MagicMock(return_value='Braille [1]')
        model = {'model': object()}
        tools = load_module('tools_under_test', 'Services/tools/agent_tools.py', {})
        coordinator = load_module('coordinator_under_test', 'Agents/coordinatorAgent.py', {
            'Services.agents.specialist_agent': dependency('engine', generate_response=generate),
            'Services.tools.agent_tools': dependency('tools', build_rag_tool=tools.build_rag_tool),
        })
        def other_tool(value: str) -> str:
            return value
        self.assertEqual(coordinator.coordinator_agent(model, ' Pergunta ', extra_tools=[other_tool]), 'Resposta')
        kwargs = generate.call_args.kwargs
        self.assertEqual(kwargs['prompt'], 'Pergunta')
        self.assertEqual(kwargs['tools'][1], other_tool)
        with patch.dict(sys.modules, {'Agents.ragAgent': dependency('rag', rag_agent=rag)}):
            self.assertEqual(kwargs['tools'][0](query=' Braille? '), 'Braille [1]')
        rag.assert_called_once_with(model, 'Pergunta')

    def test_chat_route_uses_coordinator_and_keeps_rag_endpoint(self):
        coordinator, rag = MagicMock(return_value='Coordenado'), MagicMock(return_value='Especialista')
        chat_id = '59e2b4b1-89ec-4fb6-b897-24cd93276613'
        store = MagicMock()
        store.create_chat.return_value = chat_id
        store.snapshot.return_value = ([], 0)
        with patch.object(sys, 'path', [str(ROOT), *sys.path]), patch.dict(sys.modules, {
            'Agents.coordinatorAgent': dependency('coordinator', coordinator_agent=coordinator),
            'Agents.ragAgent': dependency('rag', rag_agent=rag),
        }):
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
            from Routes.llmRoute import router
            from Controllers.LLM import coordinatorController
            from Controllers.LLM import chatHistoryController
            app = FastAPI()
            app.state.llm = {'model': 'test'}
            app.include_router(router, prefix='/llm')
            with patch.object(coordinatorController, 'ChatStore', return_value=store), patch.object(coordinatorController, 'encrypted', return_value='hashed-ip'), patch.object(chatHistoryController, 'ChatStore', return_value=store), patch.object(chatHistoryController, 'encrypted', return_value='hashed-ip') as hash_ip, TestClient(app) as client:
                store.list_chats.return_value = [{'chat_id': chat_id, 'last_question': 'Última pergunta', 'response': 'Não deve sair'}]
                listing = client.get('/llm/chats?user_id=outro-usuario', headers={'X-Forwarded-For': '1.2.3.4'})
                self.assertEqual(listing.status_code, 200)
                self.assertEqual(listing.json(), [{'chat_id': chat_id, 'last_question': 'Última pergunta'}])
                hash_ip.assert_called_once_with('testclient')
                store.list_chats.assert_called_once_with('hashed-ip')
                store.list_chats.return_value = []
                self.assertEqual(client.get('/llm/chats').json(), [])
                self.assertEqual(client.post('/llm/chats').status_code, 405)
                messages = [{'role': 'user' if index % 2 == 0 else 'assistant', 'content': f'Mensagem {index}'} for index in range(32)]
                messages[0]['url'] = 'https://www.puc-campinas.edu.br/'
                store.snapshot.return_value = (messages, 32)
                conversation = client.get(f'/llm/chats/{chat_id}?user_id=outro')
                self.assertEqual(conversation.status_code, 200)
                self.assertEqual(conversation.json(), {'chat_id': chat_id, 'messages': messages})
                store.snapshot.assert_called_with('hashed-ip', chat_id, limit=None)
                self.assertEqual(client.get('/llm/chats/id-invalido').status_code, 422)
                store.snapshot.return_value = ([], 0)
                self.assertEqual(client.get(f'/llm/chats/{chat_id}').json()['messages'], [])
                store.snapshot.side_effect = chatHistoryController.ChatNotFoundError()
                self.assertEqual(client.get(f'/llm/chats/{chat_id}').status_code, 404)
                store.snapshot.side_effect = chatHistoryController.RedisError()
                with self.assertLogs(chatHistoryController.__name__, level='ERROR'):
                    self.assertEqual(client.get(f'/llm/chats/{chat_id}').status_code, 503)
                store.snapshot.side_effect = None
                response = client.post('/llm/chat', json={'search': ' oi '})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {'chat_id': chat_id, 'query': 'oi', 'response': 'Coordenado'})
                self.assertEqual(client.post('/llm/rag', json={'search': 'SBI?'}).json()['response'], 'Especialista')
                self.assertEqual(client.post('/llm/chat', json={}).status_code, 422)
                self.assertEqual(client.post('/llm/chat', json={'search': '   '}).status_code, 422)
            coordinator.assert_called_once_with(llm=app.state.llm, query='oi', history=[])

    def test_rag_tool_passes_page_location_to_specialist(self):
        tools = load_module('location_tools', 'Services/tools/agent_tools.py', {})
        rag = MagicMock(return_value='Página SI')
        history = [{'role': 'user', 'content': 'SI'}]
        with patch.dict(sys.modules, {'Agents.ragAgent': dependency('rag', rag_agent=rag)}):
            tool = tools.build_rag_tool({}, history=history, current_url='https://www.puc-campinas.edu.br/')
            self.assertEqual(tool(query='Qual curso?'), 'Página SI')
        rag.assert_called_once_with({}, 'Qual curso?', history=history, current_url='https://www.puc-campinas.edu.br/')


if __name__ == '__main__':
    unittest.main()
