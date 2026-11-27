import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path, dependencies):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, 'path', [str(ROOT), *sys.path]), patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module


def dependency(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.collection = MagicMock()
        self.embedding = MagicMock()
        self.collection.count.return_value = 5
        self.collection.get.return_value = {'ids': [], 'documents': [], 'metadatas': []}
        self.search = load_module('search_under_test', 'Services/orm/search.py', {
            'Database.chromaDb.chroma': dependency('chroma', collection=self.collection),
            'Models.embedding_model': dependency('embedding', embedding_model=self.embedding),
        })

    def results(self, texts, distances=None):
        self.collection.query.return_value = {
            'ids': [[str(i) for i in range(len(texts))]],
            'documents': [texts], 'metadatas': [[{'source': 'manual.pdf', 'page': 20}] * len(texts)],
            'distances': [distances or [0.2] * len(texts)],
        }

    def test_rejects_original_irrelevant_context(self):
        self.results(['Documento para realizar as provas do vestibular',
                      'Documento de identificação para entrar na sala da prova',
                      'Dados do representante legal para cadastro de estágio'])
        self.assertEqual(self.search.search_evidence('Quais documentos preciso para realizar a matrícula?'), [])

    def test_deduplicates_and_keeps_sources(self):
        self.results(['Documentos para matrícula: RG e histórico.',
                      'Documentos para matrícula: RG e histórico.',
                      'Estágio e bolsas de estudo'])
        evidence = self.search.search_evidence('Documentos para matrícula')
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]['page'], 20)
        self.assertEqual(evidence[0]['source'], 'manual.pdf')

    def test_lexical_search_recovers_missing_semantic_candidate(self):
        self.results(['Estágio e bolsas de estudo'])
        self.collection.get.return_value = {
            'ids': ['list'], 'documents': ['Documentos para matrícula: RG e histórico.'],
            'metadatas': [{'source': 'edital.pdf', 'page': 42}],
        }
        self.assertEqual(self.search.search_evidence('Documentos para matrícula')[0]['page'], 42)

    def test_broad_institutional_query(self):

        self.results([
            "Vestibular da PUC-Campinas: processo seletivo."
        ])

        evidence = self.search.search_evidence(
            "Como funciona o vestibular da PUC-Campinas?"
        )

        self.assertEqual(len(evidence), 1)

    def test_requested_year_is_respected(self):

        self.results([
            "Matrícula do Vestibular 2027."
        ])

        self.assertEqual(
            self.search.search_evidence(
                "Matrícula do Vestibular 2040"
            ),
            []
        )

        self.assertEqual(
            len(self.search.search_evidence(
                "Matrícula do Vestibular 2027"
            )),
            1
        )

    def test_empty_collection_does_not_embed(self):
        self.collection.count.return_value = 0
        self.assertEqual(self.search.search_evidence('matrícula'), [])
        self.embedding.encode.assert_not_called()

    def test_rejects_distant_partial_match(self):
        self.results(['Matrícula e bolsas de estudo'], [0.99])
        self.assertEqual(self.search.search_evidence('documentos matrícula graduação'), [])


class RagTests(unittest.TestCase):
    def setUp(self):
        self.search = MagicMock()
        self.generate = MagicMock()
        self.rag = load_module('rag_under_test', 'Agents/ragAgent.py', {
            'Services.orm.search': dependency('search', search_evidence=self.search),
            'Services.agents.specialist_agent': dependency('agent', generate_response=self.generate),
        })
        self.rag.search_evidence = self.search

    def test_no_evidence_blocks_generation(self):
        self.search.return_value = []
        self.generate.return_value = 'Resposta original.'
        report = self.rag.diagnose_rag({}, 'documentos?')
        self.assertEqual(
            report['response'],
            self.rag.INSUFFICIENT_EVIDENCE
        )
        self.assertEqual(
            report['status'],
            'insufficient_evidence'
        )
        self.assertIsNone(report['raw_response'])
        self.generate.assert_not_called()

    def test_generated_response_and_system_rules_are_preserved(self):
        self.search.return_value = [{'source': 'manual.pdf', 'page': 42, 'text': 'RG e histórico'}]
        self.generate.return_value = 'RG e histórico. [1]'
        answer = self.rag.diagnose_rag({}, 'documentos?')['response']
        self.assertEqual(answer, 'RG e histórico. [1]')
        args = self.generate.call_args.kwargs
        self.assertEqual(args['tools'], [])
        self.assertIn('NÃO são documentos para matrícula', args['system_prompt'])
        self.assertIn('RG e histórico', args['prompt'])

    def test_correct_sbi_answer_without_citation_is_not_discarded(self):
        import json
        texts = json.loads((ROOT / 'tests/fixtures/sbi.json').read_text(encoding='utf-8-sig'))
        self.search.return_value = [{'source': 'guia.pdf', 'page': 12, 'text': texts[0]}]
        self.generate.return_value = 'Sim. Há livros em Braille e programas como NVDA.'
        report = self.rag.diagnose_rag({}, 'A SBI oferece inclusão visual?')
        self.assertEqual(report['status'], 'answered')
        self.assertIn('Sim. Há livros em Braille', report['response'])
        self.assertNotIn('Fontes', report['response'])
        self.assertEqual(report['raw_response'], self.generate.return_value)

    def test_invalid_citation_is_reported_without_fabricating_source(self):
        self.search.return_value = [{'source': 'guia.pdf', 'page': 12, 'text': 'Braille'}]
        self.generate.return_value = 'Sim, Braille. [9]'
        report = self.rag.diagnose_rag({}, 'Braille?')
        self.assertEqual(report['status'], 'answered')
        self.assertIn('[9]', report['response'])
        self.assertIn('[9]', report['raw_response'])

    def test_unverified_url_is_blocked_in_diagnostic(self):

        evidence = [{
            "source": "guia.pdf",
            "page": 12,
            "text": "Informações institucionais."
        }]

        self.generate.return_value = (
            "Acesse " + "https:" + "//inventado.example/"
        )

        report = self.rag.diagnose_rag(
            {}, "Onde acesso?", evidence
        )

        self.assertEqual(
            report["status"],
            "unverified_url"
        )

        self.assertEqual(
            report["response"],
            self.rag.UNVERIFIED_URL_RESPONSE
        )

        self.assertEqual(
            report["raw_response"],
            self.generate.return_value
        )

    def test_model_abstention_and_empty_output_have_distinct_statuses(self):
        evidence = [{'source': 'guia.pdf', 'page': 12, 'text': 'Braille'}]
        for text, status in [('Não encontrei informações suficientes para responder.', 'answered'), ('', 'empty_generation')]:
            self.generate.return_value = text
            self.assertEqual(self.rag.diagnose_rag({}, 'Braille?', evidence)['status'], status)


class IngestionTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'manual.pdf'
        self.path.write_bytes(b'test-pdf')
        self.collection = MagicMock()
        self.collection.get.return_value = {'ids': []}
        self.embed = MagicMock(return_value=[[0.1, 0.2]])
        self.pages = MagicMock(return_value=[(42, 'Documentos para matrícula: RG.')])
        self.ingestion = load_module('ingestion_under_test', 'Services/orm/ingestion.py', {
            'Services.orm.pdfReader': dependency('pdf', read_pdf_pages=self.pages),
            'Services.orm.chunker': dependency('chunker', split_text=lambda text: [text]),
            'Services.orm.embedding': dependency('embedding', generate_embeddings=self.embed),
            'Database.chromaDb.chroma': dependency('chroma', collection=self.collection),
        })

    def test_migrates_old_chunks_only_after_writing_new_chunks(self):
        import hashlib
        prefix = hashlib.sha256(str(self.path.resolve()).encode()).hexdigest()[:16]
        old_id = prefix + '_old_0'
        self.collection.get.return_value = {'ids': [old_id]}
        self.ingestion.process_pdf(self.path)
        written = self.collection.upsert.call_args.kwargs
        self.assertEqual(written['metadatas'][0]['page'], 42)
        self.assertEqual(written['metadatas'][0]['source'], 'manual.pdf')
        self.assertIn('_v2_', written['ids'][0])
        calls = [call[0] for call in self.collection.method_calls]
        self.assertLess(calls.index('upsert'), calls.index('delete'))
        self.collection.delete.assert_called_once_with(ids=[old_id])
        self.collection.reset_mock()
        self.collection.get.return_value = {'ids': written['ids']}
        self.ingestion.process_pdf(self.path)
        self.collection.upsert.assert_not_called()
        self.collection.delete.assert_not_called()

    def test_embedding_failure_preserves_previous_index(self):
        self.embed.side_effect = RuntimeError('embedding failed')
        with self.assertRaises(RuntimeError):
            self.ingestion.process_pdf(self.path)
        self.collection.delete.assert_not_called()
        self.collection.upsert.assert_not_called()

    def test_identical_pdf_is_ingested_once(self):
        copy = self.path.with_name('copy.pdf')
        copy.write_bytes(self.path.read_bytes())
        self.ingestion.load_pdfs = lambda: [self.path, copy]
        self.ingestion.PDF_FOLDER = self.path.parent
        self.ingestion.process_pdf = MagicMock()
        self.ingestion.initialize_knowledge_base()
        self.ingestion.process_pdf.assert_called_once()


if __name__ == '__main__':
    unittest.main()
